//! What the editor offers while you type.
//!
//! gpui-component's input already owns the whole completion surface — the
//! popup under the caret, its scroll, the arrow keys, `enter` to accept and
//! `escape` to dismiss — behind one trait with two required methods. None of
//! it needs a language server; `lsp_types` is only the vocabulary. So this
//! module is the part that is dbdelve's: which identifiers are worth offering
//! for the text in front of the cursor.
//!
//! Every object offered comes from the catalog the explorer already loaded, so
//! the popup can never name a relation the tree does not show; the rest is the
//! dialect's keywords and built-in functions and values. Nothing here
//! runs SQL and nothing here edits the buffer: an accepted row is a
//! [`TextEdit`] over the word being typed, and the input applies it.
//!
//! **This is a lexer, not a parser.** Which identifiers make sense at a
//! caret is decided by the last keyword before it and by the relations named
//! in the statement, both found by scanning tokens. A parse tree would be the
//! principled answer and is the wrong tool here: half-typed SQL is a parse
//! error by definition — `SELECT * FROM ` is exactly the text a user most
//! wants completed and exactly the text `tree_sitter_sequel` returns an
//! `ERROR` node for. The one job a scan genuinely cannot do is know it is
//! inside a string literal, and the parse's suppression scan
//! ([`sql::Parse::suppressed`]) does that job on its own.

use std::{
    borrow::Cow,
    collections::HashMap,
    ops::Range,
    sync::{Arc, Mutex, OnceLock, PoisonError},
};

use gpui::{App, AppContext as _, Result, Task, WeakEntity, Window};
use gpui_component::input::{CompletionProvider, Rope, RopeExt};
use lsp_types::{
    CompletionContext, CompletionItem, CompletionItemKind, CompletionResponse, CompletionTextEdit,
    Documentation, MarkupContent, MarkupKind, TextEdit,
};

use crate::{
    Workspace,
    db::{
        Builtin, BuiltinKind, Catalog, Engine, ForeignKey, Reference, RelationKind, Structure,
        Syntax,
    },
    fuzzy::{Fuzzy, Query},
    mql::{self, Completing},
    redis_cmd, sql,
};

/// What is known about one relation's columns.
///
/// The catalog does not carry them. Fetching every column of every relation at
/// connect was the first implementation and it does not scale: on a large
/// schema that is a multi-million-row result, buffered whole by the driver
/// before dbdelve sees a row, then held for the life of the connection and
/// duplicated into this provider's snapshot -- all of it paid before the user
/// has typed anything, and most of it for relations they will never mention.
///
/// So a relation's columns are fetched when a statement first names it, and
/// kept. What a session holds is bounded by what it actually wrote about.
#[derive(Clone, Debug)]
pub enum ColumnState {
    /// Asked for, not back yet, and how many attempts have already failed.
    /// Held so a relation is asked about once rather than on every keystroke
    /// until it lands.
    Loading(u8),
    Loaded(Described),
    /// That many attempts have failed. Retried while it is under
    /// [`FETCH_ATTEMPTS`], and then not again for this connection.
    ///
    /// Neither extreme is right here. Never retrying means one blip -- or one
    /// statement timeout, which bounds dbdelve's own catalog queries too -- kills
    /// completion for that relation silently, for the rest of the session.
    /// Always retrying means a describe per keystroke, and every one of them
    /// queues on the connection mutex behind the last, so a slow failure would
    /// freeze the profile rather than degrade it.
    Failed(u8),
}

/// What a describe told completion about one relation: its columns, and the
/// foreign keys a `JOIN` onto it or from it would follow.
#[derive(Clone, Debug, Default)]
pub struct Described {
    pub columns: Vec<String>,
    pub foreign_keys: Vec<ForeignKey>,
    /// Empty when the describe did not ask who points back, which it only
    /// does while joins are being suggested.
    pub referenced_by: Vec<Reference>,
}

impl From<&Structure> for Described {
    fn from(structure: &Structure) -> Self {
        Self {
            columns: structure
                .columns
                .iter()
                .map(|column| column.name.clone())
                .collect(),
            foreign_keys: structure.foreign_keys.clone(),
            referenced_by: structure.referenced_by.clone(),
        }
    }
}

/// How many times a relation's columns are asked for before dbdelve stops.
const FETCH_ATTEMPTS: u8 = 3;

/// How many relations' columns may be asked for at once. Each describe queues
/// on the session's one lock, so more in flight are no sooner back and only
/// hold more of the blocking pool, which a name in a few hundred schemas would
/// otherwise fill. One left over is asked for on a later keystroke.
const FETCHES_IN_FLIGHT: usize = 4;

/// Columns by schema and relation, shared between the provider that reads
/// them -- on the background executor, mid-decision -- and the workspace that
/// fills them, on the main thread when a describe lands. A std `Mutex` whose
/// critical sections are lookups and inserts, never a whole decision and
/// never an await.
pub type ColumnCache = Arc<Mutex<HashMap<(String, String), ColumnState>>>;

/// Schema and relation pairs whose columns a completion wanted and the cache
/// did not hold.
type Wanted = Vec<(String, String)>;

/// How many rows the popup is allowed. The list is already ranked, so the tail
/// of a 4,000-column database is not what anyone is scrolling for.
const MAX_ITEMS: usize = 50;

/// The keywords worth completing: the ones that carry a statement's shape.
///
/// Deliberately not the full reserved-word list of three engines. A completion
/// list is a menu, and a menu of four hundred keywords is a wall — these are
/// the ones long enough to be worth not typing and common enough to be worth
/// ranking above a column that happens to fuzzy-match.
const KEYWORDS: &[&str] = &[
    "AND",
    "AS",
    "ASC",
    "BETWEEN",
    "BY",
    "CASE",
    "CAST",
    "COALESCE",
    "CROSS JOIN",
    "DESC",
    "DISTINCT",
    "ELSE",
    "END",
    "EXCEPT",
    "EXISTS",
    "FALSE",
    "FROM",
    "FULL JOIN",
    "GROUP BY",
    "HAVING",
    "ILIKE",
    "IN",
    "INNER JOIN",
    "INSERT INTO",
    "INTERSECT",
    "IS NOT NULL",
    "IS NULL",
    "JOIN",
    "LATERAL",
    "LEFT JOIN",
    "LIKE",
    "LIMIT",
    "NOT",
    "NULL",
    "NULLS FIRST",
    "NULLS LAST",
    "OFFSET",
    "ON",
    "OR",
    "ORDER BY",
    "OVER",
    "PARTITION BY",
    "RETURNING",
    "RIGHT JOIN",
    "SELECT",
    "SET",
    "THEN",
    "TRUE",
    "UNION",
    "UNION ALL",
    "UPDATE",
    "USING",
    "VALUES",
    "WHEN",
    "WHERE",
    "WITH",
];

/// Keywords after which an identifier names a relation.
const SOURCE_KEYWORDS: &[&str] = &["from", "join", "update", "into", "table"];

/// A completion the popup can offer, before it is ranked.
///
/// Cloned only for the few that survive ranking; the pool's copy of a
/// candidate is what a keystroke scores against.
#[derive(Clone)]
struct Candidate {
    label: String,
    /// The grey text beside the label — the schema a relation is in, the
    /// relation a column belongs to. What tells two `id`s apart.
    detail: Option<String>,
    kind: CompletionItemKind,
    /// Whether accepting it writes the relation under the schema in `detail`.
    /// A flag rather than the name, which is spelled only for the few that
    /// survive ranking; ranking reads the label alone either way.
    qualified: bool,
    /// The built-in it offers, whose call and documentation are spelled, like
    /// a qualified name, only for the few that survive ranking.
    builtin: Option<&'static Builtin>,
    /// The alias a relation is written with, where one belongs.
    alias: Option<String>,
    /// The condition that joins it to a relation the statement already
    /// holds, spelled with that alias.
    on: Option<String>,
}

impl Candidate {
    fn new(label: impl Into<String>, detail: Option<String>, kind: CompletionItemKind) -> Self {
        Self {
            label: label.into(),
            detail,
            kind,
            qualified: false,
            builtin: None,
            alias: None,
            on: None,
        }
    }

    fn qualified(self, qualified: bool) -> Self {
        Self { qualified, ..self }
    }

    fn builtin(builtin: &'static Builtin) -> Self {
        let (detail, kind) = match builtin.kind {
            BuiltinKind::Function => ("function", CompletionItemKind::FUNCTION),
            BuiltinKind::Aggregate => ("aggregate", CompletionItemKind::FUNCTION),
            BuiltinKind::Window => ("window", CompletionItemKind::FUNCTION),
            BuiltinKind::Value => ("value", CompletionItemKind::CONSTANT),
        };
        Self {
            builtin: Some(builtin),
            ..Self::new(builtin.name, Some(detail.to_string()), kind)
        }
    }

    /// What accepting it writes into the buffer.
    ///
    /// A function is written up to its open parenthesis, ready for the first
    /// argument, and closed as well when it takes none.
    fn written(&self, engine: Engine) -> String {
        if let Some(builtin) = self.builtin {
            return match (builtin.kind, builtin.args) {
                (BuiltinKind::Value, _) => builtin.name.to_string(),
                (_, "") => format!("{}()", builtin.name),
                _ => format!("{}(", builtin.name),
            };
        }
        let name = match (self.qualified, &self.detail) {
            (true, Some(schema)) => format!(
                "{}.{}",
                engine.quote_identifier_if_needed(schema),
                engine.quote_identifier_if_needed(&self.label)
            ),
            _ => self.label.clone(),
        };
        format!("{name}{}", self.aliasing())
    }

    /// The alias and join condition that follow the name, in the buffer and
    /// in the popup alike, so the row shows what accepting it writes.
    fn aliasing(&self) -> String {
        match (&self.alias, &self.on) {
            (Some(alias), Some(on)) => format!(" {alias} ON {on}"),
            (Some(alias), None) => format!(" {alias}"),
            (None, Some(on)) => format!(" ON {on}"),
            (None, None) => String::new(),
        }
    }

    /// A built-in's sentence over its signature.
    ///
    /// The sentence leads because a menu too narrow to sit beside its
    /// documentation shows only the first line of it. The fence names no
    /// language: a language's tag is drawn over the end of a wrapped line, and
    /// its highlighter colours `with` in `timestamp with time zone` as a
    /// keyword.
    fn documentation(&self) -> Option<Documentation> {
        let builtin = self.builtin?;
        let signature = match builtin.kind {
            BuiltinKind::Value => format!("{} → {}", builtin.name, builtin.returns),
            _ => format!("{}({}) → {}", builtin.name, builtin.args, builtin.returns),
        };
        Some(Documentation::MarkupContent(MarkupContent {
            kind: MarkupKind::Markdown,
            value: format!("{}\n\n```\n{signature}\n```", builtin.doc),
        }))
    }
}

/// The completion source for a connected profile's editor.
///
/// Holds a snapshot rather than a live handle. The catalog is replaced whole
/// when it reloads, and so is this — see `Workspace::install_completions`.
/// Everything the catalog decides — the pool of names to rank, the parse that
/// finds the statement the caret sits in — is therefore that snapshot's own,
/// and a reload replaces it with the catalog and not after it.
///
/// Cloned per request: the trait hands the provider `&self` and the heavy
/// work has to move to the background executor, which takes a `Send` future,
/// so a clone goes into the task instead. The clone shares every cache — the
/// fields are `Arc`s and `Copy` values — so the popup and the filter bars it
/// is installed beside all decide from one parse and one pool.
#[derive(Clone)]
pub struct SchemaCompletions {
    engine: Engine,
    catalog: Arc<Catalog>,
    columns: ColumnCache,
    qualify: bool,
    /// Whether a relation completed after `FROM` or `JOIN` is given an alias,
    /// and one after `JOIN` the condition its foreign keys imply.
    alias: bool,
    /// Who to ask when a relation's columns are wanted and not yet held.
    ///
    /// `None` only in this module's own tests, which drive `items`
    /// against a pre-filled cache and never reach the fetch.
    workspace: Option<WeakEntity<Workspace>>,
    /// The matcher scores through `&mut self` and the trait hands us `&self`.
    /// One matcher reused rather than one per keystroke, as the palette does.
    /// Behind a lock because the decision that scores through it runs on the
    /// background executor; the guard lives only across the ranking.
    fuzzy: Arc<Mutex<Fuzzy>>,
    /// The buffer's parse, kept from the last keystroke that needed it. Its
    /// incremental state is the point — a keystroke reparses from its edit —
    /// so one parse is shared and serialized behind this lock, never copied
    /// per request, and the lock spans the reads that must see one text.
    /// Poison is deliberately ignored: a panicked compute costs one request,
    /// not the editor — the next request takes the guard and `statement_at`
    /// re-derives whatever the panic left from the text it holds.
    parsed: Arc<Mutex<sql::Parse>>,
    /// The catalog's half of the pool, built for the first keystroke that
    /// ranks it and never after: the catalog cannot move under it, because a
    /// catalog that reloads replaces this provider whole. Two requests racing
    /// to build it serialize in `get_or_init` — the second waits for the
    /// first, so it is built exactly once.
    pool: Arc<OnceLock<Pool>>,
}

impl SchemaCompletions {
    pub fn new(
        engine: Engine,
        catalog: Arc<Catalog>,
        columns: ColumnCache,
        qualify: bool,
        alias: bool,
        workspace: WeakEntity<Workspace>,
    ) -> Self {
        Self {
            engine,
            catalog,
            columns,
            qualify,
            alias,
            workspace: Some(workspace),
            fuzzy: Arc::new(Mutex::new(Fuzzy::new())),
            parsed: Arc::new(Mutex::new(sql::Parse::new())),
            pool: Arc::new(OnceLock::new()),
        }
    }

    /// The pool, built for the first keystroke that ranks from it.
    fn pool(&self) -> &Pool {
        self.pool
            .get_or_init(|| Pool::new(self.engine, &self.catalog))
    }

    /// The columns of a relation named anywhere in the catalog, as far as they
    /// are known, recording into `wanted` any this cache has never been asked
    /// for.
    ///
    /// Unqualified, so two schemas with a `users` each contribute both sets.
    /// Offering a column that turns out to belong to the other schema's table
    /// is a worse answer than offering nothing only if the user cannot see
    /// which — which is what `detail` is for.
    ///
    /// A name the catalog does not list is never recorded. The fetch would ask
    /// the server to describe something it has already said does not exist,
    /// once per keystroke, for as long as the typo is on screen.
    fn columns_of(&self, relation: &str, wanted: &mut Vec<(String, String)>) -> Vec<Candidate> {
        self.columns_of_except(relation, None, wanted)
    }

    /// [`Self::columns_of`], less the one relation `except` names, whose
    /// columns the caller has already offered.
    fn columns_of_except(
        &self,
        relation: &str,
        except: Option<&(String, String)>,
        wanted: &mut Wanted,
    ) -> Vec<Candidate> {
        let mut candidates = Vec::new();
        for schema in &self.catalog.schemas {
            for named in schema
                .relations
                .iter()
                .filter(|named| named.name.eq_ignore_ascii_case(relation))
            {
                let key = (schema.name.clone(), named.name.clone());
                if except != Some(&key) {
                    candidates.extend(self.columns_at(key, wanted));
                }
            }
        }
        candidates
    }

    /// The columns of exactly this schema's relation, for a caller that knows
    /// which one it means. Nothing for one the catalog does not list, for the
    /// reason [`Self::columns_of`] gives.
    fn columns_in(
        &self,
        (schema, relation): &(String, String),
        wanted: &mut Wanted,
    ) -> Vec<Candidate> {
        let listed = self.catalog.schemas.iter().any(|listed| {
            &listed.name == schema && listed.relations.iter().any(|named| &named.name == relation)
        });
        match listed {
            true => self.columns_at((schema.clone(), relation.clone()), wanted),
            false => Vec::new(),
        }
    }

    /// The columns held for one relation, recording it into `wanted` when
    /// they are not.
    ///
    /// Recorded once however often a statement names it: each entry is a
    /// describe on the wire, and a second would reset the first's count of
    /// failed attempts.
    fn columns_at(&self, key: (String, String), wanted: &mut Wanted) -> Vec<Candidate> {
        let relation = key.1.clone();
        self.described(key, wanted, |described| {
            described
                .columns
                .iter()
                .map(|column| {
                    Candidate::new(column, Some(relation.clone()), CompletionItemKind::FIELD)
                })
                .collect()
        })
        .unwrap_or_default()
    }

    /// What is held about one relation, read through `read`, recording it
    /// into `wanted` when nothing is.
    ///
    /// `read` runs under the lock, as the borrow it replaces did, and none of
    /// the readers reach back into the cache, so the section stays short.
    fn described<T>(
        &self,
        key: (String, String),
        wanted: &mut Wanted,
        read: impl FnOnce(&Described) -> T,
    ) -> Option<T> {
        let mut want = |key| {
            if !wanted.contains(&key) {
                wanted.push(key);
            }
        };
        let columns = self.columns.lock().unwrap_or_else(PoisonError::into_inner);
        match columns.get(&key) {
            Some(ColumnState::Loaded(described)) => return Some(read(described)),
            // Failed, but not yet often enough to give up on.
            Some(ColumnState::Failed(attempts)) if *attempts < FETCH_ATTEMPTS => want(key),
            // In flight, or asked for as often as it is going to be.
            Some(ColumnState::Loading(_) | ColumnState::Failed(_)) => {}
            None => want(key),
        }
        None
    }

    /// The catalog's own listing of a relation a foreign key names. A key
    /// that names no schema finds its relation when only one has that name.
    fn listed(&self, schema: &str, relation: &str) -> Option<(String, String)> {
        let mut named = self
            .catalog
            .schemas
            .iter()
            .filter(|listed| schema.is_empty() || listed.name == schema)
            .flat_map(|listed| {
                listed
                    .relations
                    .iter()
                    .filter(|named| named.name == relation)
                    .map(|named| (listed.name.clone(), named.name.clone()))
            });
        let first = named.next()?;
        named.next().is_none().then_some(first)
    }

    /// The relations a foreign key joins to the ones `head`, the statement up
    /// to the caret, already holds, in either direction.
    ///
    /// Not one the statement holds already: after `orders o JOIN order_items
    /// oi` every key between the two would offer the other one again. A key
    /// from a relation to itself is the exception, being the only way to ask
    /// for a self-join.
    fn joins(&self, head: &str, wanted: &mut Wanted) -> Vec<Join> {
        let code = code(head);
        let ctes = ctes(&code);
        // A CTE hides the table it shares a name with, and has none of its
        // keys.
        let sources = sources(&reach(&code))
            .into_iter()
            .filter(|source| {
                source.schema.is_some()
                    || !ctes
                        .iter()
                        .any(|cte| cte.eq_ignore_ascii_case(&source.relation))
            })
            .collect::<Vec<_>>();
        let held = |(schema, relation): &(String, String)| {
            sources.iter().any(|source| {
                source.relation.eq_ignore_ascii_case(relation)
                    && source
                        .schema
                        .as_ref()
                        .is_none_or(|named| named.eq_ignore_ascii_case(schema))
            })
        };
        let mut joins: Vec<Join> = Vec::new();
        for source in &sources {
            let keys = self
                .catalog
                .schemas
                .iter()
                .filter(|schema| {
                    source
                        .schema
                        .as_ref()
                        .is_none_or(|named| named.eq_ignore_ascii_case(&schema.name))
                })
                .flat_map(|schema| {
                    schema
                        .relations
                        .iter()
                        .filter(|named| named.name.eq_ignore_ascii_case(&source.relation))
                        .map(|named| (schema.name.clone(), named.name.clone()))
                })
                .collect::<Vec<_>>();
            for key in keys {
                let found = self.described(key.clone(), wanted, |described| {
                    let mut found = foreign_keys(&described.foreign_keys);
                    found.extend(described.referenced_by.iter().map(|reference| {
                        (
                            (reference.schema.clone(), reference.table.clone()),
                            vec![(
                                reference.column.clone(),
                                reference.referenced_column.clone(),
                            )],
                        )
                    }));
                    found
                });
                for ((schema, relation), columns) in found.unwrap_or_default() {
                    let Some(target) = self.listed(&schema, &relation) else {
                        continue;
                    };
                    if held(&target) && target != key {
                        continue;
                    }
                    let join = Join {
                        target,
                        onto: source.alias.clone().unwrap_or(source.relation.clone()),
                        columns,
                    };
                    if !joins.contains(&join) {
                        joins.push(join);
                    }
                }
            }
        }
        joins
    }

    /// `relations` as the caret wants them written: each with an alias where
    /// one belongs, led after `JOIN` by the ones a foreign key joins to the
    /// relations already held, with the condition.
    fn aliased(
        &self,
        around: &Around,
        relations: Vec<Candidate>,
        scope: Option<&(String, String)>,
        wanted: &mut Wanted,
    ) -> Vec<Candidate> {
        // A filter bar's scope is a relation the text never names, so there
        // is nothing an alias there could be used for.
        if !self.alias || scope.is_some() || !around.takes_alias() {
            return relations;
        }
        // Without the word being typed, which `FROM o` would otherwise bind
        // as a relation called `o` and cost `orders` its alias.
        let rest = code(&format!("{} {}", around.head, around.tail));
        let taken = bindings(&rest)
            .into_iter()
            .map(|(binding, _)| binding)
            .collect::<Vec<_>>();
        let here = sources(&reach(&code(around.head)));
        // A relation the statement already reaches into by name keeps it:
        // `SELECT orders.id FROM orders o` no longer has an `orders`.
        let alias = |relation: &str| {
            let named = tokens(&rest).into_iter().any(|token| {
                token
                    .split_once('.')
                    .is_some_and(|(qualifier, _)| qualifier.eq_ignore_ascii_case(relation))
            });
            // Unless it is held already: a second `employees` beside the
            // first needs a name of its own.
            let held = here
                .iter()
                .any(|held| held.relation.eq_ignore_ascii_case(relation));
            (!named || held).then(|| alias_for(relation, &taken))
        };

        let mut candidates = Vec::new();
        if around.takes_condition() {
            for join in self.joins(around.head, wanted) {
                let Some(relation) = relations.iter().find(|relation| {
                    relation.label == join.target.1
                        && relation.detail.as_ref() == Some(&join.target.0)
                }) else {
                    continue;
                };
                let alias = alias(&relation.label);
                let name = alias.as_deref().unwrap_or(&relation.label);
                let on = join
                    .columns
                    .iter()
                    .map(|(theirs, ours)| format!("{name}.{theirs} = {}.{ours}", join.onto))
                    .collect::<Vec<_>>()
                    .join(" AND ");
                candidates.push(Candidate {
                    alias,
                    on: Some(on),
                    ..Candidate::new(&relation.label, relation.detail.clone(), relation.kind)
                        .qualified(relation.qualified)
                });
            }
        }
        candidates.extend(relations.into_iter().map(|relation| Candidate {
            alias: alias(&relation.label),
            ..relation
        }));
        candidates
    }

    fn relations_in(&self, schema_name: &str) -> Vec<Candidate> {
        self.catalog
            .schemas
            .iter()
            .filter(|schema| schema.name.eq_ignore_ascii_case(schema_name))
            .flat_map(|schema| {
                schema.relations.iter().map(|rel| {
                    Candidate::new(
                        &rel.name,
                        Some(schema.name.clone()),
                        relation_kind(rel.kind),
                    )
                })
            })
            .collect()
    }

    fn all_relations(&self, qualify: bool) -> Vec<Candidate> {
        catalog_relations(&self.catalog, qualify)
    }

    fn schema_names(&self) -> Vec<Candidate> {
        self.catalog
            .schemas
            .iter()
            .map(|schema| Candidate::new(&schema.name, None, CompletionItemKind::MODULE))
            .collect()
    }

    /// What to offer, in preference order, before ranking narrows it.
    ///
    /// Order matters twice over: the scorer's ties are broken by it, and an
    /// empty query scores everything equally, so this order *is* the list the
    /// user sees after a `.`.
    ///
    /// The candidates that are the catalog's own — keywords, relations,
    /// routines, built-ins — are borrowed from the pool and never copied to be
    /// ranked; the statement's own are built here and owned.
    fn candidates<'a>(
        &self,
        around: &Around,
        qualifier: Option<&str>,
        scope: Option<&(String, String)>,
        wanted: &mut Vec<(String, String)>,
        pool: &'a Pool,
    ) -> Vec<Cow<'a, Candidate>> {
        let Around {
            statement, before, ..
        } = *around;
        if let Some(qualifier) = qualifier {
            // `x.` — `x` is an alias if the statement bound one, otherwise the
            // relation or schema it looks like. An alias wins: a query that
            // says `FROM orders o` has made `o` mean something here, and a
            // table elsewhere in the database called `o` has not.
            let relation = bindings(statement)
                .into_iter()
                .find(|(binding, _)| binding.eq_ignore_ascii_case(qualifier))
                .map(|(_, relation)| relation)
                .unwrap_or_else(|| qualifier.to_string());

            let columns = self.columns_of(&relation, wanted);
            if !columns.is_empty() {
                return columns.into_iter().map(Cow::Owned).collect();
            }
            // Nothing held for it yet, which may only mean a fetch that has not
            // landed. Offering the schema's relations is the right answer when
            // the qualifier is a schema, and harmless when it is not: an
            // unknown name matches nothing.
            return self
                .aliased(around, self.relations_in(qualifier), scope, wanted)
                .into_iter()
                .map(Cow::Owned)
                .collect();
        }

        // A qualifier `qualifier_before` cannot read, `"audit".` say, is still
        // one, and qualifying after it would write a second.
        let qualify = self.qualify && !before.ends_with('.');
        if expects_relation(before) {
            let mut candidates = self
                .aliased(around, self.all_relations(qualify), scope, wanted)
                .into_iter()
                .map(Cow::Owned)
                .collect::<Vec<_>>();
            candidates.extend(self.schema_names().into_iter().map(Cow::Owned));
            return candidates;
        }

        // Columns of the relations this statement actually names come first:
        // in a statement about `accounts`, `email` is a better guess than
        // every other `email` in the database, and than `EXISTS`. The scope's
        // columns lead them, as the relation the text is about without naming
        // it, and are not offered again when the text names it too.
        let mut candidates: Vec<Cow<'a, Candidate>> = scope
            .map(|scope| self.columns_in(scope, wanted))
            .unwrap_or_default()
            .into_iter()
            .map(Cow::Owned)
            .collect();
        candidates.extend(
            bindings(statement)
                .into_iter()
                .map(|(_, relation)| relation)
                .fold(Vec::new(), |mut relations, relation| {
                    if !relations.iter().any(|held| held == &relation) {
                        relations.push(relation);
                    }
                    relations
                })
                .iter()
                .flat_map(|relation| self.columns_of_except(relation, scope, wanted))
                .map(Cow::Owned),
        );
        candidates.extend(pool.keywords.iter().map(Cow::Borrowed));
        candidates.extend(pool.relations[qualify as usize].iter().map(Cow::Borrowed));
        candidates.extend(pool.routines.iter().map(Cow::Borrowed));
        // Last, so on a tie the user's own objects win and a dialect's long
        // list can't push them past the cap. Not after an unreadable
        // qualifier: `"o".date_trunc(` is never a call.
        if !before.ends_with('.') {
            candidates.extend(pool.builtins.iter().map(Cow::Borrowed));
        }
        candidates
    }

    /// [`Self::candidates`] for a MongoDB buffer: what `mql::completing` says
    /// the word is, from the catalog and the field names sampled for the
    /// collection's structure.
    fn mongo_candidates(
        &self,
        text: &str,
        word: Range<usize>,
        scope: Option<&(String, String)>,
        wanted: &mut Wanted,
    ) -> Vec<Candidate> {
        fn methods(names: impl Iterator<Item = &'static str>) -> Vec<Candidate> {
            names
                .map(|name| Candidate::new(name, None, CompletionItemKind::METHOD))
                .collect()
        }
        match mql::completing(text, word.start) {
            None => Vec::new(),
            // `db.` already names the database a collection is in.
            Some(Completing::Database) => {
                let mut candidates = self.all_relations(false);
                candidates.extend(methods(mql::DbMethod::ALL.iter().map(|m| m.name())));
                candidates
            }
            Some(Completing::Collection) => methods(mql::Method::ALL.iter().map(|m| m.name())),
            Some(Completing::Cursor) => methods(mql::CursorMethod::ALL.iter().map(|m| m.name())),
            // As in SQL, nothing typed is not a question.
            Some(Completing::Document { .. }) if word.is_empty() => Vec::new(),
            Some(Completing::Document { collection }) => {
                let mut candidates = match scope {
                    Some(scope) => self.columns_in(scope, wanted),
                    None => collection
                        .map(|collection| self.columns_of(&collection, wanted))
                        .unwrap_or_default(),
                };
                candidates.extend(
                    mql::OPERATORS.iter().map(|operator| {
                        Candidate::new(*operator, None, CompletionItemKind::OPERATOR)
                    }),
                );
                candidates
            }
        }
    }

    /// The whole decision, from buffer text to the rows the popup shows.
    ///
    /// Separate from [`CompletionProvider::completions`] and taking `&str`
    /// rather than a `Rope` so the tests can reach it without a window.
    /// Returns the rows to show, and the relations whose columns are wanted and
    /// not yet held — which [`Self::respond`] admits, on the main thread,
    /// because admission needs the `App`.
    #[cfg(test)]
    fn items(&self, sql: &str, offset: usize) -> (Vec<(Range<usize>, Candidate)>, Wanted) {
        self.items_in(sql, offset, None)
    }

    /// The whole decision, from buffer text to the rows the popup shows, for
    /// text about one relation: its `(schema, relation)` is `scope`, whose
    /// columns are offered first. `items` is the tests' window onto this.
    fn items_in(
        &self,
        sql: &str,
        offset: usize,
        scope: Option<&(String, String)>,
    ) -> (Vec<(Range<usize>, Candidate)>, Wanted) {
        let mut wanted = Vec::new();
        let offset = offset.min(sql.len());
        if !sql.is_char_boundary(offset) {
            return (Vec::new(), wanted);
        }

        let word = word_before(sql, offset);
        match self.engine.syntax() {
            Syntax::Sql => {}
            Syntax::Redis => {
                let (word, candidates) = redis_candidates(sql, offset);
                let ranked = rank(
                    candidates.into_iter().map(Cow::Owned).collect(),
                    &sql[word.clone()],
                    &mut self.fuzzy.lock().unwrap_or_else(PoisonError::into_inner),
                )
                .into_iter()
                .map(|candidate| (word.clone(), candidate))
                .collect();
                return (ranked, wanted);
            }
            Syntax::Mongo => {
                let candidates = self.mongo_candidates(sql, word.clone(), scope, &mut wanted);
                let ranked = rank(
                    candidates.into_iter().map(Cow::Owned).collect(),
                    &sql[word.clone()],
                    &mut self.fuzzy.lock().unwrap_or_else(PoisonError::into_inner),
                )
                .into_iter()
                .map(|candidate| (word.clone(), candidate))
                .collect();
                return (ranked, wanted);
            }
        }
        let prefix = &sql[word.clone()];
        let qualifier = qualifier_before(sql, word.start);

        // Nothing typed and nothing to qualify is not a question. Opening the
        // whole catalog on every space would put a popup over the buffer for
        // most of the time anyone is writing in it. Suppression waits until
        // past this point for the same reason: a keystroke with no answer
        // owes no scan for one.
        if prefix.is_empty() && qualifier.is_none() {
            return (Vec::new(), wanted);
        }

        // The parse is the provider's own, kept from the last keystroke that
        // asked, and brought up to date by this one: a keystroke reparses from
        // its edit, not from the top of the buffer (`sql::Parse`). It syncs
        // here, before the suppression scan reads its resume point, and both
        // reads hold the lock together: they must describe the same text, and
        // the editor and the filter bar drive one provider, so a concurrent
        // request could otherwise move the parse between the two. The ranking
        // after this runs unlocked -- it reads the text, the catalog and the
        // caches, not the parse.
        let (statement, inside) = {
            let mut parsed = self.parsed.lock().unwrap_or_else(PoisonError::into_inner);
            let statement = parsed
                .statement_at(sql, offset)
                .filter(|range| range.contains(&word.start) || range.end >= word.start)
                .unwrap_or(0..sql.len());
            let inside = parsed.suppressed(word.start);
            (statement, inside)
        };

        // Silence inside a string literal or a comment, asked of the parse's
        // text -- current since the statement above -- which resumes where the
        // last keystroke's scan stopped instead of walking from byte 0.
        if inside {
            return (Vec::new(), wanted);
        }

        let around = Around {
            statement: &sql[statement.clone()],
            before: &sql[..word.start],
            head: &sql[statement.start.min(word.start)..word.start],
            tail: &sql[offset..statement.end.max(offset)],
        };

        // The pool's borrow lives through the ranking, which reads it without
        // copying it; only the rows that survive the cap are owned.
        let ranked = {
            let pool = self.pool();
            let candidates = self.candidates(&around, qualifier, scope, &mut wanted, pool);
            rank(
                candidates,
                prefix,
                &mut self.fuzzy.lock().unwrap_or_else(PoisonError::into_inner),
            )
        }
        .into_iter()
        .map(|candidate| (word.clone(), candidate))
        .collect();
        (ranked, wanted)
    }

    /// The heavy half of one completion request, run on the background
    /// executor: the buffer copy when the text moved, the parse's incremental
    /// sync, and the decision [`Self::items_in`] makes. What the keystroke's
    /// own thread pays is the probe, a rope clone and this spawn.
    fn compute(
        &self,
        rope: &Rope,
        offset: usize,
        scope: Option<&(String, String)>,
    ) -> (Vec<(Range<usize>, Candidate)>, Wanted) {
        // The copy the scans read, taken under the parse's lock so a
        // concurrent request cannot sync the parse between the check and the
        // adopt: skipped when the parse already holds this text — a keystroke
        // that only re-ranks still allocates nothing — else one rope-to-
        // `String` copy, paid here and not inside the keystroke.
        //
        // ponytail: the rope arrives as a rope and every scan reads a `str`,
        // so a changed keystroke still pays two buffer copies — the rope to a
        // `String`, then the `String` into the `Arc<str>` the parse holds
        // (`Parse::adopt`), which stable Rust cannot fold into one: the
        // refcount header must precede the data in the same allocation, so
        // the `String`'s buffer is not reusable. They used to sit inside the
        // keystroke; round two of the typing-lag fix moved them here, where
        // they cost a background thread a few hundred microseconds instead
        // of the character its echo. Ending the copies means teaching the
        // trait the edit event: the input knows what moved and does not hand
        // it to the provider. Hold off until the trait offers one.
        let sql: Arc<str> = {
            let mut parsed = self.parsed.lock().unwrap_or_else(PoisonError::into_inner);
            if parse_holds(&parsed, rope) {
                parsed.held()
            } else {
                let sql: Arc<str> = Arc::from(rope.to_string());
                parsed.adopt(&sql);
                sql
            }
        };
        self.items_in(&sql, offset, scope)
    }
}

impl CompletionProvider for SchemaCompletions {
    /// Every edit asks, and the decision — [`Self::items_in`], running on
    /// the background executor — answers.
    ///
    /// Answering `false` here would leave an open popup standing, because the
    /// input only reconsiders the menu when the provider is consulted. An
    /// empty answer from `completions` hides it, so there is one place that
    /// decides whether a popup belongs and it is the one that knows.
    fn is_completion_trigger(&self, _: usize, _: &str, _: &mut App) -> bool {
        true
    }

    fn completions(
        &self,
        text: &Rope,
        offset: usize,
        _: CompletionContext,
        _: &mut Window,
        cx: &mut App,
    ) -> Task<Result<CompletionResponse>> {
        // Most keystrokes do not ask a question: after a space, a newline or
        // a paren the menu has nothing to offer, and two bytes off the rope
        // say so, before the buffer is copied for the scans that would answer
        // it. MongoDB's scan reads the text it wraps and cannot decide from
        // the caret's neighbourhood, so it keeps the copy.
        if self.engine.syntax() == Syntax::Sql && nothing_typed(text, offset) {
            return Task::ready(Ok(CompletionResponse::Array(Vec::new())));
        }
        // Everything the answer costs — the buffer copy, the parse, the
        // scans, the ranking — runs on the background executor, and this
        // returns before any of it has run: the character echoes first. A
        // request the engine replaces before its compute finishes is
        // cancelled with the task that awaited it (`context_menu_task` is
        // dropped per request), so no guard of this provider's own is needed
        // to keep a superseded answer from landing. Admission — the one part
        // that needs the app, to ask the workspace for a describe — waits for
        // the compute on the main thread, in `respond`.
        let provider = self.clone();
        let rope = text.clone();
        let compute = cx.background_spawn(async move {
            let computed = provider.compute(&rope, offset, None);
            (rope, computed)
        });
        let provider = self.clone();
        cx.spawn(async move |cx| {
            let (rope, computed) = compute.await;
            Ok(cx.update(|cx| provider.respond(&rope, computed, cx)))
        })
    }
}

impl SchemaCompletions {
    /// Ask for the columns `computed` wanted, and assemble the rows the popup
    /// shows as edits over `text`, the rope the ranges are offsets into.
    ///
    /// On the main thread rather than in the background compute, because the
    /// fetch admission asks the workspace to describe — the one part of a
    /// request that needs the app.
    fn respond(
        &self,
        text: &Rope,
        (items, wanted): (Vec<(Range<usize>, Candidate)>, Wanted),
        cx: &mut App,
    ) -> CompletionResponse {
        // Marked before the request so a second keystroke arriving while the
        // first fetch is in flight does not ask again. The compute that
        // produced `wanted` read the cache before this ran, though, so a key
        // already `Loading` here is a concurrent request's — skipped, or two
        // describes would go out for one relation.
        for key in self.admitted(wanted) {
            let before = self
                .columns
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(&key)
                .cloned();
            if matches!(before, Some(ColumnState::Loading(_))) {
                continue;
            }
            let attempts = match &before {
                Some(ColumnState::Failed(attempts)) => *attempts,
                _ => 0,
            };
            self.columns
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(key.clone(), ColumnState::Loading(attempts));
            let asked = self.workspace.as_ref().is_some_and(|workspace| {
                let (schema, relation) = key.clone();
                workspace
                    .update(cx, |workspace, cx| {
                        workspace.load_completion_columns(schema, relation, cx)
                    })
                    .unwrap_or(false)
            });
            // A `Loading` nothing will answer would hold a place in flight
            // for the rest of the session.
            if !asked {
                let mut columns = self.columns.lock().unwrap_or_else(PoisonError::into_inner);
                match before {
                    Some(state) => columns.insert(key, state),
                    None => columns.remove(&key),
                };
            }
        }

        // The popup does not reopen by itself when a fetch lands: the input
        // reconsiders the menu on an edit and nothing else. The next keystroke
        // shows the columns, which for a name being typed is the very next one.
        let items = items
            .into_iter()
            .map(|(word, candidate)| CompletionItem {
                // The edit is given explicitly rather than left to the input's
                // own guess at where the word started. That guess is taken on
                // the first keystroke of a run and not revised, so a word typed
                // one character at a time replaces the wrong span -- and the
                // wrong span here is a corrupted statement.
                text_edit: Some(CompletionTextEdit::Edit(TextEdit {
                    range: lsp_types::Range {
                        start: text.offset_to_position(word.start),
                        end: text.offset_to_position(word.end),
                    },
                    new_text: candidate.written(self.engine),
                })),
                documentation: candidate.documentation(),
                label: format!("{}{}", candidate.label, candidate.aliasing()),
                detail: candidate.detail,
                kind: Some(candidate.kind),
                ..Default::default()
            })
            .collect::<Vec<_>>();

        CompletionResponse::Array(items)
    }

    /// As many of `wanted`, in order, as may be asked for alongside the
    /// fetches already in flight.
    fn admitted(&self, mut wanted: Wanted) -> Wanted {
        let in_flight = self
            .columns
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .filter(|state| matches!(state, ColumnState::Loading(_)))
            .count();
        wanted.truncate(FETCHES_IN_FLIGHT.saturating_sub(in_flight));
        wanted
    }
}

/// Whether the parse already holds `text`: the rope's chunks against the held
/// text, byte for byte, without building a copy to compare with. The first
/// chunk that disagrees settles it.
fn parse_holds(parse: &sql::Parse, text: &Rope) -> bool {
    let held = parse.held();
    let held: &str = &held;
    let mut read = 0;
    for chunk in text.chunks() {
        if held.get(read..read + chunk.len()) != Some(chunk) {
            return false;
        }
        read += chunk.len();
    }
    read == held.len()
}

/// The completion source for a raw filter bar on one relation's tab.
///
/// A raw filter is only a `WHERE`, so on its own it names no relation and
/// [`SchemaCompletions`] would have no columns to offer. The tab's relation is
/// handed over as the scope instead, and everything else -- the column fetch,
/// the keywords, the silence inside a literal -- is the query editor's own.
///
/// The relation is never written into the text the scan reads: a name holding
/// a `'`, a `--` or a space would read as a literal, a comment or two names,
/// and silence the completion it was there to feed.
///
/// The schema is behind an `Arc` because the bar's heavy half runs on the
/// background executor too, and that executor takes a `Send` future; the bar
/// itself never crosses a thread.
pub struct FilterCompletions {
    schema: Arc<SchemaCompletions>,
    relation: (String, String),
    /// What the text is read behind. A MongoDB filter is a document, which
    /// the scan only reads as one inside a call.
    prefix: &'static str,
}

impl FilterCompletions {
    /// `None` on an engine whose raw filter the schema cannot complete: Redis's
    /// is a `MATCH` glob over key names.
    pub fn new(
        syntax: Syntax,
        schema: Arc<SchemaCompletions>,
        relation: (String, String),
    ) -> Option<Self> {
        let prefix = match syntax {
            Syntax::Sql => "",
            // As `mql::browse::raw_filter` wraps it.
            Syntax::Mongo => "db.c.find(\n",
            Syntax::Redis => return None,
        };
        Some(Self {
            schema,
            relation,
            prefix,
        })
    }

    /// [`SchemaCompletions::compute`] for the bar's own text, with the ranges
    /// back in that text. The tests' way in; the trait goes through
    /// [`CompletionProvider::completions`], which runs the same decision on
    /// the background executor.
    #[cfg(test)]
    fn items(&self, text: &str, offset: usize) -> (Vec<(Range<usize>, Candidate)>, Wanted) {
        let shift = self.prefix.len();
        let (items, wanted) = self.schema.items_in(
            &format!("{}{text}", self.prefix),
            shift + offset,
            Some(&self.relation),
        );
        (bar_items(items, shift), wanted)
    }
}

/// The computed rows back into the bar's own text: the prefixed document's
/// ranges shifted back, and qualifying off -- an editor setting, and the bar
/// is not the editor.
fn bar_items(
    items: Vec<(Range<usize>, Candidate)>,
    shift: usize,
) -> Vec<(Range<usize>, Candidate)> {
    items
        .into_iter()
        .filter(|(word, _)| word.start >= shift)
        .map(|(word, candidate)| {
            (
                word.start - shift..word.end - shift,
                Candidate {
                    qualified: false,
                    ..candidate
                },
            )
        })
        .collect()
}

impl CompletionProvider for FilterCompletions {
    /// For the reason [`SchemaCompletions`] gives.
    fn is_completion_trigger(&self, _: usize, _: &str, _: &mut App) -> bool {
        true
    }

    fn completions(
        &self,
        text: &Rope,
        offset: usize,
        _: CompletionContext,
        _: &mut Window,
        cx: &mut App,
    ) -> Task<Result<CompletionResponse>> {
        // The bar's text becomes the parse's document with the prefix in
        // front, so the editor's own scan reads it; the ranges come back
        // prefixed and are shifted back before the response is assembled over
        // the bar's rope. The copy and the decision run on the background
        // executor, as the editor's request does; admission stays on the main
        // thread for the reason [`SchemaCompletions::respond`] gives.
        let schema = Arc::clone(&self.schema);
        let relation = self.relation.clone();
        let shift = self.prefix.len();
        let prefix = self.prefix;
        let rope = text.clone();
        let compute = cx.background_spawn(async move {
            let document = Rope::from(format!("{}{}", prefix, rope));
            let (items, wanted) = schema.compute(&document, shift + offset, Some(&relation));
            (rope, bar_items(items, shift), wanted)
        });
        let schema = Arc::clone(&self.schema);
        cx.spawn(async move |cx| {
            let (rope, items, wanted) = compute.await;
            Ok(cx.update(|cx| schema.respond(&rope, (items, wanted), cx)))
        })
    }
}

fn relation_kind(kind: RelationKind) -> CompletionItemKind {
    match kind {
        RelationKind::Table
        | RelationKind::PartitionedTable
        | RelationKind::ForeignTable
        | RelationKind::Dictionary => CompletionItemKind::STRUCT,
        RelationKind::View | RelationKind::MaterializedView => CompletionItemKind::INTERFACE,
    }
}

/// Every relation in the catalog, `qualify` marking that accepting one writes
/// the schema it is in.
fn catalog_relations(catalog: &Catalog, qualify: bool) -> Vec<Candidate> {
    catalog
        .schemas
        .iter()
        .flat_map(|schema| {
            schema.relations.iter().map(move |rel| Candidate {
                qualified: qualify,
                ..Candidate::new(
                    &rel.name,
                    Some(schema.name.clone()),
                    relation_kind(rel.kind),
                )
            })
        })
        .collect()
}

/// The catalog's half of the pool: the names a keystroke ranks whether or not
/// the statement says anything yet, built once and borrowed from.
///
/// What a statement decides — the columns of the relations it names, the
/// aliases it has taken — stays per keystroke; this never changes while the
/// provider lives, and a catalog that reloads replaces the provider whole
/// (`Workspace::install_completions`), pool and all.
struct Pool {
    /// The keywords worth completing, less the ones the dialect has as a
    /// built-in, which are offered once, as the built-in.
    keywords: Vec<Candidate>,
    /// The relations, with `qualified` off and on: which a keystroke wants is
    /// the text in front of the caret's decision, and ranking does not read it.
    relations: [Vec<Candidate>; 2],
    routines: Vec<Candidate>,
    builtins: Vec<Candidate>,
}

impl Pool {
    fn new(engine: Engine, catalog: &Catalog) -> Self {
        let builtins = engine.builtins();
        Self {
            keywords: KEYWORDS
                .iter()
                .filter(|keyword| {
                    !builtins
                        .iter()
                        .any(|builtin| builtin.name.eq_ignore_ascii_case(keyword))
                })
                .map(|keyword| Candidate::new(*keyword, None, CompletionItemKind::KEYWORD))
                .collect(),
            relations: [
                catalog_relations(catalog, false),
                catalog_relations(catalog, true),
            ],
            routines: catalog
                .schemas
                .iter()
                .flat_map(|schema| {
                    schema.routines.iter().map(|routine| {
                        Candidate::new(
                            &routine.name,
                            Some(schema.name.clone()),
                            CompletionItemKind::FUNCTION,
                        )
                    })
                })
                .collect(),
            builtins: builtins.iter().map(Candidate::builtin).collect(),
        }
    }
}

/// Score every candidate against what has been typed, best first.
///
/// The same matcher the palette uses, for the same reason: a fuzzy match is
/// what people expect from a list that narrows as they type.
///
/// Not [`Fuzzy::rank`]: that gives a tie to the shorter label, and here a tie
/// keeps the order [`SchemaCompletions::candidates`] built, which puts the
/// statement's own columns ahead of keywords. The pool's candidates are
/// borrowed and the statement's owned, so only the rows that survive the cap
/// are cloned out of either.
fn rank(candidates: Vec<Cow<'_, Candidate>>, prefix: &str, fuzzy: &mut Fuzzy) -> Vec<Candidate> {
    let query = Query::new(prefix);
    let mut scored = candidates
        .into_iter()
        .filter_map(|candidate| Some((fuzzy.score(&query, &candidate.label)?, candidate)))
        .collect::<Vec<_>>();
    // A join the schema declares leads whatever its score: it has matched
    // what was typed like every other row, and there are only ever a few.
    scored.sort_by_key(|(score, candidate)| std::cmp::Reverse((candidate.on.is_some(), *score)));
    scored.truncate(MAX_ITEMS);
    scored
        .into_iter()
        .map(|(_, candidate)| candidate.into_owned())
        .collect()
}

fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'$'
}

/// The command name being typed, when the caret is in the first word of its
/// line, and every name to rank against it. A name may hold a dot
/// (`FT.CREATE`), which `word_before` would cut at.
fn redis_candidates(text: &str, offset: usize) -> (Range<usize>, Vec<Candidate>) {
    let line = text[..offset].rfind('\n').map_or(0, |newline| newline + 1);
    let typed = &text[line..offset];
    let start = line + typed.len() - typed.trim_start().len();
    let word = start..offset;
    let first_word = !text[word.clone()].contains(char::is_whitespace);
    if !first_word || text[word.clone()].starts_with('#') || word.is_empty() {
        return (word, Vec::new());
    }
    let candidates = redis_cmd::names()
        .map(|(name, tag)| {
            Candidate::new(name, tag.map(String::from), CompletionItemKind::FUNCTION)
        })
        .collect();
    (word, candidates)
}

/// The identifier being typed, as a range ending at the cursor. Empty when the
/// cursor does not follow one.
///
/// Byte-wise, and safe to be: every byte of a multi-byte character is `>= 0x80`
/// and no ASCII test matches one, so a scan can never stop mid-character.
fn word_before(sql: &str, offset: usize) -> Range<usize> {
    let bytes = sql.as_bytes();
    let mut start = offset;
    while start > 0 && is_ident_byte(bytes[start - 1]) {
        start -= 1;
    }
    start..offset
}

/// The identifier before the `.` that precedes the word being typed — the `o`
/// of `o.`, the `public` of `public.acc`.
fn qualifier_before(sql: &str, word_start: usize) -> Option<&str> {
    let bytes = sql.as_bytes();
    if word_start == 0 || bytes[word_start - 1] != b'.' {
        return None;
    }

    let dot = word_start - 1;
    let mut start = dot;
    while start > 0 && is_ident_byte(bytes[start - 1]) {
        start -= 1;
    }
    (start < dot).then(|| &sql[start..dot])
}

/// Whether nothing is typed and nothing qualifies at the caret -- where
/// the decision has no answer and returns none -- read
/// straight off the rope rather than out of a copy of the buffer.
///
/// The two bytes in front of the caret are the whole question: a word ends at
/// the caret exactly when the byte before it is an identifier byte
/// ([`word_before`]), and a qualifier when that byte is a `.` with an
/// identifier byte in front of it ([`qualifier_before`]). Everything else the
/// scan would look at -- the statement, the catalog, the strings and comments
/// -- only matters once one of the two is there.
fn nothing_typed(text: &Rope, offset: usize) -> bool {
    let offset = offset.min(text.len());
    if !text.is_char_boundary(offset) {
        return true;
    }
    let byte = |at: usize| text.bytes_at(at).next();
    match offset.checked_sub(1).and_then(byte) {
        None => true,
        Some(byte) if is_ident_byte(byte) => false,
        // A qualifier ends here when an identifier byte sits in front of the
        // dot; a `.` at the very start has none in front of it.
        Some(b'.') => offset < 2 || !byte(offset - 2).is_some_and(is_ident_byte),
        Some(_) => true,
    }
}

/// The word-shaped tokens of a statement, in order. Dots are kept inside a
/// token so `public.accounts` stays one name rather than becoming two.
fn tokens(sql: &str) -> Vec<&str> {
    let bytes = sql.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;

    while index < bytes.len() {
        if is_ident_byte(bytes[index]) || bytes[index] == b'.' {
            let start = index;
            while index < bytes.len() && (is_ident_byte(bytes[index]) || bytes[index] == b'.') {
                index += 1;
            }
            tokens.push(&sql[start..index]);
        } else {
            index += 1;
        }
    }

    tokens
}

fn is_keyword(token: &str) -> bool {
    KEYWORDS.iter().any(|keyword| {
        keyword
            .split(' ')
            .any(|word| word.eq_ignore_ascii_case(token))
    }) || SOURCE_KEYWORDS
        .iter()
        .any(|keyword| keyword.eq_ignore_ascii_case(token))
}

/// The text around the word being completed.
#[derive(Clone, Copy)]
struct Around<'a> {
    /// The statement the caret is in, whole.
    statement: &'a str,
    /// The buffer up to the word.
    before: &'a str,
    /// The statement up to the word, and from the caret on.
    head: &'a str,
    tail: &'a str,
}

impl Around<'_> {
    /// The keyword the name being typed follows, past a schema typed in
    /// front of it.
    fn source(&self) -> Option<&str> {
        let tokens = tokens(self.before);
        let name = tokens.len().checked_sub(1)?;
        match tokens[name].ends_with('.') {
            true => tokens.get(name.checked_sub(1)?).copied(),
            false => Some(tokens[name]),
        }
    }

    /// Whether a relation written here can carry an alias: after `FROM` or
    /// `JOIN`, with nothing behind the caret that already is one.
    ///
    /// Not `DELETE FROM`, where SQLite and SQL Server take no alias, nor the
    /// other source keywords, where none does.
    fn takes_alias(&self) -> bool {
        let tokens = tokens(self.before);
        let after_delete = tokens
            .iter()
            .rev()
            .skip_while(|token| token.ends_with('.'))
            .nth(1)
            .is_some_and(|token| token.eq_ignore_ascii_case("delete"));
        let follows = self.source().is_some_and(|keyword| {
            ["from", "join"]
                .iter()
                .any(|k| keyword.eq_ignore_ascii_case(k))
        });
        follows && !after_delete && self.next().is_none_or(|next| next != Next::Name)
    }

    /// Whether a join condition can be written here as well: after `JOIN`,
    /// in front of no `ON` of the statement's own. Not after `CROSS JOIN` or
    /// `NATURAL JOIN`, which take none.
    fn takes_condition(&self) -> bool {
        let unconditional = tokens(self.before)
            .iter()
            .rev()
            .skip_while(|token| token.ends_with('.'))
            .skip(1)
            .take_while(|token| is_join_modifier(token))
            .any(|token| {
                ["cross", "natural"]
                    .iter()
                    .any(|k| token.eq_ignore_ascii_case(k))
            });
        self.source()
            .is_some_and(|keyword| keyword.eq_ignore_ascii_case("join"))
            && !unconditional
            && self.next().is_none_or(|next| next != Next::Condition)
    }

    /// What the text behind the caret opens with, when it is anything.
    fn next(&self) -> Option<Next> {
        let rest = self.tail.trim_start();
        let first = *rest.as_bytes().first()?;
        // The caret is inside a word, or in front of a quoted name or a
        // call's arguments: whatever is written here is not the whole name.
        let inside = rest.len() == self.tail.len() && is_ident_byte(first);
        // Nor is it known past a comment, which may sit in front of an alias.
        let comment = rest.starts_with("--") || rest.starts_with("/*");
        if inside || comment || matches!(first, b'"' | b'`' | b'[' | b'(' | b'.') {
            return Some(Next::Name);
        }
        if !is_ident_byte(first) {
            return Some(Next::Other);
        }
        let token = tokens(rest).into_iter().next()?;
        Some(match token {
            token
                if ["on", "using"]
                    .iter()
                    .any(|k| token.eq_ignore_ascii_case(k)) =>
            {
                Next::Condition
            }
            token if is_keyword(token) && !token.eq_ignore_ascii_case("as") => Next::Other,
            _ => Next::Name,
        })
    }
}

/// A word that may stand between a join's kind and its `JOIN`.
fn is_join_modifier(token: &str) -> bool {
    [
        "cross", "natural", "inner", "left", "right", "full", "outer",
    ]
    .iter()
    .any(|modifier| token.eq_ignore_ascii_case(modifier))
}

/// The part of `head` whose relations a condition at its end can name: less
/// every parenthesis already closed, and from the one the caret is inside.
/// A subquery's relations, or a CTE's, are out of reach of the query around
/// it, and the reverse.
fn reach(head: &str) -> String {
    let mut reach = String::new();
    let mut opened = Vec::new();
    for character in head.chars() {
        match character {
            '(' => opened.push(reach.len()),
            ')' => reach.truncate(opened.pop().unwrap_or(0)),
            _ => {}
        }
        // A space where a group closed, so the words around it stay two.
        reach.push(if character == ')' { ' ' } else { character });
    }
    let mut reach = match opened.last() {
        Some(&open) => reach.split_off(open),
        None => reach,
    };
    // The query on the other side of a `UNION` is as far out of reach.
    let operand = words(&reach)
        .into_iter()
        .filter(|word| {
            ["union", "intersect", "except"]
                .iter()
                .any(|operator| reach[word.clone()].eq_ignore_ascii_case(operator))
        })
        .map(|word| word.end)
        .next_back();
    match operand {
        Some(start) => reach.split_off(start),
        None => reach,
    }
}

/// `sql` with each string literal and comment blanked, so that the words in
/// one are not read as the statement's own.
fn code(sql: &str) -> String {
    let bytes = sql.as_bytes();
    let mut code = String::with_capacity(sql.len());
    let (mut index, mut copied) = (0, 0);
    while index < bytes.len() {
        let end = match bytes[index] {
            b'\'' => {
                let mut end = index + 1;
                loop {
                    match bytes.get(end) {
                        None => break end,
                        Some(b'\'') if bytes.get(end + 1) == Some(&b'\'') => end += 2,
                        Some(b'\'') => break end + 1,
                        Some(_) => end += 1,
                    }
                }
            }
            b'-' if bytes.get(index + 1) == Some(&b'-') => sql[index..]
                .find('\n')
                .map_or(bytes.len(), |offset| index + offset),
            b'/' if bytes.get(index + 1) == Some(&b'*') => sql[index + 2..]
                .find("*/")
                .map_or(bytes.len(), |offset| index + 2 + offset + 2),
            _ => {
                index += 1;
                continue;
            }
        };
        code.push_str(&sql[copied..index]);
        code.push(' ');
        (index, copied) = (end, end);
    }
    code.push_str(&sql[copied..]);
    code
}

/// Where each word of `sql` sits.
fn words(sql: &str) -> Vec<Range<usize>> {
    let bytes = sql.as_bytes();
    let mut words = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        let start = index;
        while index < bytes.len() && is_ident_byte(bytes[index]) {
            index += 1;
        }
        match index > start {
            true => words.push(start..index),
            false => index += 1,
        }
    }
    words
}

/// The names a `WITH` gives its queries: each word followed by `AS (`.
fn ctes(code: &str) -> Vec<&str> {
    let words = words(code);
    let mut ctes = Vec::new();
    for (index, word) in words.iter().enumerate().skip(1) {
        if !code[word.clone()].eq_ignore_ascii_case("as") {
            continue;
        }
        // `AS NOT MATERIALIZED (` is the same query under a hint.
        let body = words[index + 1..]
            .iter()
            .take_while(|hint| {
                ["not", "materialized"]
                    .iter()
                    .any(|known| code[(*hint).clone()].eq_ignore_ascii_case(known))
            })
            .last()
            .unwrap_or(word)
            .end;
        if code[body..].trim_start().starts_with('(') {
            ctes.push(&code[words[index - 1].clone()]);
        }
    }
    ctes
}

/// What follows the caret, as far as an alias written at it is concerned.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Next {
    /// More of the name, or an alias the statement already gives it.
    Name,
    /// The join's own `ON` or `USING`.
    Condition,
    Other,
}

/// A relation a foreign key joins to one the statement holds.
#[derive(PartialEq, Eq)]
struct Join {
    /// Its schema and name, as the catalog spells them.
    target: (String, String),
    /// The name the statement knows the held relation by.
    onto: String,
    /// Its column and the held relation's, for each column of the key.
    columns: Vec<(String, String)>,
}

/// A relation's foreign keys, by the relation each points at, as that
/// relation's column and this one's.
///
/// ponytail: a [`ForeignKey`] is one column of a key and does not say which
/// key, so two keys onto one relation and one key of two columns arrive
/// looking alike. No key names a column twice, so a column met again opens
/// the next key: `created_by` and `updated_by` both onto `users.id` are two
/// joins. That leans on a key's columns arriving together, and reads two
/// single-column keys onto different columns of one relation as one key of
/// two -- carry the constraint's name on `ForeignKey` if either is ever met.
fn foreign_keys(keys: &[ForeignKey]) -> Vec<(JoinTarget, Vec<(String, String)>)> {
    let mut grouped: Vec<(JoinTarget, Vec<(String, String)>)> = Vec::new();
    for key in keys {
        let target = (key.referenced_schema.clone(), key.referenced_table.clone());
        let columns = (key.referenced_column.clone(), key.column.clone());
        match grouped.iter_mut().find(|(held, _)| held == &target) {
            Some((_, held)) => held.push(columns),
            None => grouped.push((target, vec![columns])),
        }
    }
    grouped
        .into_iter()
        .flat_map(|(target, columns)| {
            let mut keys: Vec<Vec<(String, String)>> = Vec::new();
            for (theirs, ours) in columns {
                let repeats = keys.last().is_none_or(|key| {
                    key.iter().any(|(held_theirs, held_ours)| {
                        held_theirs == &theirs || held_ours == &ours
                    })
                });
                match keys.last_mut().filter(|_| !repeats) {
                    Some(key) => key.push((theirs, ours)),
                    None => keys.push(vec![(theirs, ours)]),
                }
            }
            keys.into_iter().map(move |key| (target.clone(), key))
        })
        .collect()
}

/// The schema and relation a foreign key points at, as the key spells them.
type JoinTarget = (String, String);

/// Words an alias cannot be without quoting, beyond the keywords completion
/// offers. Short ones only: an alias is a relation's initials.
const RESERVED_ALIASES: &[&str] = &[
    "add", "all", "any", "at", "do", "for", "go", "if", "int", "key", "no", "of", "on", "or",
    "row", "set", "to", "top", "use",
];

/// The alias a relation is given: the initials of the words in its name,
/// numbered when the statement already uses them or they spell a keyword.
fn alias_for(relation: &str, taken: &[String]) -> String {
    let mut initials = String::new();
    let mut previous: Option<char> = None;
    for character in relation.chars() {
        let starts_word = previous.is_none_or(|previous| {
            !previous.is_ascii_alphanumeric()
                || (previous.is_ascii_lowercase() && character.is_ascii_uppercase())
        });
        if starts_word && character.is_ascii_alphanumeric() {
            initials.push(character.to_ascii_lowercase());
        }
        previous = Some(character);
    }
    // A name with no letters to take, or one whose first is a digit, which
    // cannot open an identifier.
    if !initials.starts_with(|first: char| first.is_ascii_lowercase()) {
        initials.insert(0, 't');
    }

    let free = |alias: &str| {
        !is_keyword(alias)
            && !RESERVED_ALIASES.contains(&alias)
            && !taken.iter().any(|held| held.eq_ignore_ascii_case(alias))
    };
    if free(&initials) {
        return initials;
    }
    (2..)
        .map(|number| format!("{initials}{number}"))
        .find(|alias| free(alias))
        .unwrap_or(initials)
}

/// Whether the caret sits where a relation name belongs — straight after
/// `FROM`, `JOIN`, `UPDATE`, `INSERT INTO` or `TABLE`.
fn expects_relation(before: &str) -> bool {
    tokens(before).last().is_some_and(|last| {
        SOURCE_KEYWORDS
            .iter()
            .any(|keyword| last.eq_ignore_ascii_case(keyword))
    })
}

/// Every name this statement binds to a relation: the relation's own name, and
/// its alias where it has one.
///
/// Returned as pairs rather than a map because the same alias can legitimately
/// be bound twice in one buffer and the first binding is the one in scope.
fn bindings(statement: &str) -> Vec<(String, String)> {
    sources(statement)
        .into_iter()
        .flat_map(|source| {
            let alias = source.alias.map(|alias| (alias, source.relation.clone()));
            std::iter::once((source.relation.clone(), source.relation)).chain(alias)
        })
        .collect()
}

/// A relation a statement reads from or writes to, as it names it.
struct Source {
    schema: Option<String>,
    relation: String,
    alias: Option<String>,
}

/// Every relation named after a source keyword, in order.
fn sources(statement: &str) -> Vec<Source> {
    let tokens = tokens(statement);
    let mut sources = Vec::new();
    let mut index = 0;

    while index < tokens.len() {
        let is_source = SOURCE_KEYWORDS
            .iter()
            .any(|keyword| tokens[index].eq_ignore_ascii_case(keyword));

        let Some(name) = tokens.get(index + 1).filter(|_| is_source) else {
            index += 1;
            continue;
        };
        if is_keyword(name) {
            index += 1;
            continue;
        }

        // A schema-qualified name binds under its last segment: `FROM
        // public.accounts` puts `accounts.` in reach, which is what anyone
        // types next.
        let mut segments = name.rsplit('.');
        let relation = segments.next().unwrap_or(name).to_string();
        let schema = segments.next().map(str::to_string);

        let alias = if tokens
            .get(index + 2)
            .is_some_and(|token| token.eq_ignore_ascii_case("as"))
        {
            tokens.get(index + 3)
        } else {
            tokens.get(index + 2)
        };
        sources.push(Source {
            schema,
            relation,
            alias: alias
                .filter(|alias| !is_keyword(alias))
                .map(|alias| (*alias).to_string()),
        });

        index += 2;
    }

    sources
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{Engine, Relation, Routine, RoutineKind, Schema};

    /// A provider whose cache already holds every relation's columns, as it
    /// would once the statement had named them and the fetches had landed.
    ///
    /// `workspace: None` — these drive [`SchemaCompletions::items`], which
    /// reports what it wants rather than asking for it. What is *not* held is
    /// tested separately, in the two tests that build their own cache.
    fn detached(catalog: Catalog, columns: &[(&str, &str, &[&str])]) -> SchemaCompletions {
        let held = columns
            .iter()
            .map(|(schema, relation, columns)| {
                (
                    (schema.to_string(), relation.to_string()),
                    ColumnState::Loaded(Described {
                        columns: columns.iter().map(|column| column.to_string()).collect(),
                        ..Described::default()
                    }),
                )
            })
            .collect();
        SchemaCompletions {
            engine: Engine::Postgres,
            catalog: Arc::new(catalog),
            columns: Arc::new(Mutex::new(held)),
            qualify: false,
            alias: false,
            workspace: None,
            fuzzy: Arc::new(Mutex::new(Fuzzy::new())),
            parsed: Arc::new(Mutex::new(sql::Parse::new())),
            pool: Arc::new(OnceLock::new()),
        }
    }

    fn relation(name: &str) -> Relation {
        Relation {
            name: name.to_string(),
            kind: RelationKind::Table,
            partition_of: None,
            size: None,
            rows: None,
        }
    }

    const COLUMNS: &[(&str, &str, &[&str])] = &[
        ("public", "accounts", &["id", "email", "created_at"]),
        ("public", "orders", &["id", "account_id", "total"]),
        ("audit", "events", &["id", "payload"]),
    ];

    fn schemas() -> Catalog {
        Catalog {
            schemas: vec![
                Schema {
                    name: "public".to_string(),
                    relations: vec![relation("accounts"), relation("orders")],
                    routines: vec![Routine {
                        name: "recalculate_totals".to_string(),
                        kind: RoutineKind::Procedure,
                        identity_arguments: String::new(),
                        result_type: String::new(),
                        language: "plpgsql".to_string(),
                        definition: String::new(),
                    }],
                    sequences: Vec::new(),
                },
                Schema {
                    name: "audit".to_string(),
                    relations: vec![relation("events")],
                    routines: Vec::new(),
                    sequences: Vec::new(),
                },
            ],
            database: None,
        }
    }

    fn catalog() -> SchemaCompletions {
        detached(schemas(), COLUMNS)
    }

    /// The labels offered for a caret at `|`.
    fn labels(sql: &str) -> Vec<String> {
        labels_on(Engine::Postgres, sql)
    }

    /// [`labels`] on another engine's dialect.
    fn labels_on(engine: Engine, sql: &str) -> Vec<String> {
        let offset = sql.find('|').expect("the test must mark the caret");
        let sql = sql.replace('|', "");
        SchemaCompletions {
            engine,
            ..catalog()
        }
        .items(&sql, offset)
        .0
        .into_iter()
        .map(|(_, candidate)| candidate.label)
        .collect()
    }

    #[test]
    fn after_from_it_offers_relations() {
        let offered = labels("SELECT * FROM acc|");
        assert_eq!(offered.first().map(String::as_str), Some("accounts"));
    }

    #[test]
    fn after_from_it_offers_no_keywords() {
        let offered = labels("SELECT * FROM a|");
        assert!(
            !offered.iter().any(|label| label == "AND"),
            "a relation position offered a keyword: {offered:?}"
        );
    }

    #[test]
    fn it_offers_schemas_where_a_relation_belongs() {
        assert!(labels("SELECT * FROM aud|").contains(&"audit".to_string()));
    }

    #[test]
    fn a_schema_qualifier_offers_its_relations() {
        assert_eq!(labels("SELECT * FROM audit.|"), vec!["events".to_string()]);
    }

    #[test]
    fn a_relation_qualifier_offers_its_columns() {
        assert_eq!(
            labels("SELECT accounts.| FROM accounts"),
            vec![
                "id".to_string(),
                "email".to_string(),
                "created_at".to_string()
            ]
        );
    }

    #[test]
    fn an_alias_offers_the_relations_columns() {
        assert_eq!(
            labels("SELECT o.| FROM orders o"),
            vec![
                "id".to_string(),
                "account_id".to_string(),
                "total".to_string()
            ]
        );
    }

    #[test]
    fn an_as_alias_is_read_the_same_way() {
        assert_eq!(
            labels("SELECT o.| FROM orders AS o"),
            vec![
                "id".to_string(),
                "account_id".to_string(),
                "total".to_string()
            ]
        );
    }

    #[test]
    fn an_alias_beats_a_relation_of_the_same_name() {
        // `orders o` makes `o` mean orders. Nothing else may claim it.
        assert!(labels("SELECT o.| FROM orders o").contains(&"total".to_string()));
    }

    #[test]
    fn an_unknown_qualifier_offers_nothing() {
        assert!(labels("SELECT nothing.| FROM accounts").is_empty());
    }

    #[test]
    fn columns_of_the_named_relation_come_before_keywords() {
        let offered = labels("SELECT e| FROM accounts");
        let email = offered.iter().position(|label| label == "email");
        let end = offered.iter().position(|label| label == "END");
        assert!(
            email < end,
            "a column of the statement's own relation ranked below a keyword: {offered:?}"
        );
    }

    #[test]
    fn a_schema_qualified_relation_still_binds_its_columns() {
        assert!(labels("SELECT accounts.| FROM public.accounts").contains(&"email".to_string()));
    }

    #[test]
    fn nothing_typed_and_nothing_qualified_offers_nothing() {
        assert!(labels("SELECT * FROM |").is_empty());
        assert!(labels("|").is_empty());
    }

    #[test]
    fn nothing_typed_agrees_with_the_scan_of_the_copied_text() {
        // The probe decides whether the buffer is worth copying at all, so it
        // must answer exactly as the scan of that copy would. The long buffer
        // puts the caret past the rope's first chunk, where its two bytes
        // live in different ones.
        let long = format!("SELECT * FROM {}", "word ".repeat(600));
        let cases = [
            "",
            "|",
            "SELECT * FROM |",
            "SELECT * FROM acc|",
            "SELECT accounts.|",
            "SELECT (.|",
            "SELECT a.b.|",
            "\"audit\".|",
            "é|",
            "é", // no caret marker: the caret falls at text.len(), just past the character
            "SELECT é FROM t |",
            &format!("{long}|"),
            &format!("SELECT * FROM {long}wo|rd"),
        ];
        for text in cases {
            let offset = text.find('|').unwrap_or(text.len());
            let sql = text.replace('|', "");
            let word = word_before(&sql, offset);
            let qualifier = qualifier_before(&sql, word.start);
            assert_eq!(
                nothing_typed(&Rope::from(&sql[..]), offset),
                word.is_empty() && qualifier.is_none(),
                "{text:?}"
            );
        }
        // A caret inside a character is none of the loop's offsets, which are
        // all boundaries, so the early return that makes one nothing typed --
        // and the provider's answer empty -- is pinned directly.
        assert!(nothing_typed(&Rope::from("é"), 1));
    }

    #[test]
    fn the_pool_is_built_for_the_first_keystroke_that_ranks_it() {
        // Nothing to build until a keystroke asks, and a buffer that completes
        // from mql or redis commands never asks at all.
        let completions = catalog();
        assert!(completions.pool.get().is_none());
        let offered = completions.items("SELECT * FROM acc", 17).0;
        assert!(
            offered
                .iter()
                .any(|(_, candidate)| candidate.label == "accounts")
        );
        assert!(completions.pool.get().is_some());
    }

    #[test]
    fn a_pool_serves_no_catalog_but_the_one_its_provider_was_installed_with() {
        // `install_completions` replaces the provider whole when the catalog
        // reloads, so the pool is that provider's own. Were it ever shared,
        // the relation a refresh dropped would stay offered for the life of
        // the session.
        let stale = detached(schemas(), COLUMNS);
        assert!(
            stale
                .items("SELECT * FROM acc", 17)
                .0
                .iter()
                .any(|(_, candidate)| candidate.label == "accounts")
        );

        let reloaded = detached(
            Catalog {
                schemas: vec![Schema {
                    name: "public".to_string(),
                    relations: vec![relation("events")],
                    routines: Vec::new(),
                    sequences: Vec::new(),
                }],
                database: None,
            },
            &[],
        );
        let offered = reloaded.items("SELECT * FROM eve", 17).0;
        assert!(
            offered
                .iter()
                .any(|(_, candidate)| candidate.label == "events")
        );
        assert!(
            !offered
                .iter()
                .any(|(_, candidate)| candidate.label == "accounts")
        );
    }

    #[test]
    fn a_string_literal_suppresses_everything() {
        assert!(labels("SELECT * FROM accounts WHERE email = 'acc|'").is_empty());
    }

    #[test]
    fn a_closed_string_does_not_suppress_what_follows() {
        assert!(!labels("SELECT * FROM accounts WHERE email = 'x' AND ema|").is_empty());
    }

    #[test]
    fn an_escaped_quote_does_not_end_the_literal() {
        assert!(labels("SELECT * FROM accounts WHERE email = 'it''s acc|'").is_empty());
    }

    #[test]
    fn a_line_comment_suppresses_to_the_end_of_the_line() {
        assert!(labels("-- pick the acc|\nSELECT 1").is_empty());
        assert!(!labels("-- a note\nSELECT * FROM acc|").is_empty());
    }

    #[test]
    fn a_block_comment_suppresses_until_it_closes() {
        assert!(labels("/* acc| */ SELECT 1").is_empty());
        assert!(!labels("/* a note */ SELECT * FROM acc|").is_empty());
    }

    #[test]
    fn the_edit_replaces_exactly_the_word_being_typed() {
        let sql = "SELECT * FROM acc";
        let (word, _) = catalog()
            .items(sql, sql.len())
            .0
            .into_iter()
            .next()
            .expect("a relation was offered");
        assert_eq!(&sql[word], "acc");
    }

    #[test]
    fn a_qualified_edit_replaces_the_word_and_not_the_qualifier() {
        let sql = "SELECT o.tot FROM orders o";
        let offset = sql.find(" FROM").expect("the caret sits after `tot`");
        let (word, candidate) = catalog()
            .items(sql, offset)
            .0
            .into_iter()
            .next()
            .expect("a column was offered");
        assert_eq!(&sql[word], "tot");
        assert_eq!(candidate.label, "total");
    }

    /// The first label offered for a caret at `|`, and what accepting it writes.
    fn first_edit(sql: &str, qualify: bool) -> (String, String) {
        let offset = sql.find('|').expect("the test must mark the caret");
        let sql = sql.replace('|', "");
        let completions = SchemaCompletions {
            qualify,
            ..catalog()
        };
        let (_, candidate) = completions
            .items(&sql, offset)
            .0
            .into_iter()
            .next()
            .expect("something was offered");
        (
            candidate.label.clone(),
            candidate.written(completions.engine),
        )
    }

    #[test]
    fn qualifying_writes_a_relation_with_its_schema_and_ranks_it_by_its_name() {
        assert_eq!(
            first_edit("SELECT * FROM eve|", true),
            ("events".to_string(), "audit.events".to_string())
        );
    }

    #[test]
    fn qualifying_does_not_repeat_a_schema_already_typed() {
        assert_eq!(
            first_edit("SELECT * FROM audit.eve|", true),
            ("events".to_string(), "events".to_string())
        );
    }

    #[test]
    fn qualifying_does_not_repeat_a_quoted_schema_already_typed() {
        assert_eq!(
            first_edit("SELECT * FROM \"audit\".eve|", true),
            ("events".to_string(), "events".to_string())
        );
    }

    #[test]
    fn a_qualified_name_quotes_each_part_that_would_not_read_back_unquoted() {
        let completions = SchemaCompletions {
            qualify: true,
            ..detached(
                Catalog {
                    schemas: vec![Schema {
                        name: "Audit Log".to_string(),
                        relations: vec![relation("events")],
                        routines: Vec::new(),
                        sequences: Vec::new(),
                    }],
                    database: None,
                },
                &[],
            )
        };
        let (_, candidate) = completions
            .items("SELECT * FROM eve", 17)
            .0
            .into_iter()
            .next()
            .expect("a relation was offered");
        assert_eq!(
            candidate.written(completions.engine),
            "\"Audit Log\".events"
        );
    }

    #[test]
    fn a_relation_is_written_bare_unless_qualifying_is_on() {
        assert_eq!(
            first_edit("SELECT * FROM eve|", false),
            ("events".to_string(), "events".to_string())
        );
    }

    #[test]
    fn a_routine_is_offered() {
        assert!(labels("SELECT recalc|").contains(&"recalculate_totals".to_string()));
    }

    #[test]
    fn a_dialects_own_built_ins_are_offered_on_its_engine_only() {
        assert!(labels("SELECT date_tr|").contains(&"date_trunc".to_string()));
        assert!(
            !labels_on(Engine::SqlServer, "SELECT date_tr|").contains(&"date_trunc".to_string())
        );
        assert!(labels_on(Engine::SqlServer, "SELECT getd|").contains(&"GETDATE".to_string()));
        assert!(!labels("SELECT getd|").contains(&"GETDATE".to_string()));
    }

    #[test]
    fn a_built_in_is_written_as_a_call_or_a_bare_value() {
        let written = |sql: &str, label: &str| {
            let completions = catalog();
            let (_, candidate) = completions
                .items(sql, sql.len())
                .0
                .into_iter()
                .find(|(_, candidate)| candidate.label == label)
                .expect("the built-in was offered");
            assert!(candidate.documentation().is_some(), "{label}");
            candidate.written(completions.engine)
        };
        assert_eq!(written("SELECT date_trunc", "date_trunc"), "date_trunc(");
        assert_eq!(written("SELECT now", "now"), "now()");
        assert_eq!(written("SELECT current_d", "CURRENT_DATE"), "CURRENT_DATE");
    }

    #[test]
    fn current_offers_the_current_values() {
        let offered = labels("SELECT CURRENT_|");
        for value in ["CURRENT_DATE", "CURRENT_TIMESTAMP", "CURRENT_USER"] {
            assert!(offered.contains(&value.to_string()), "{offered:?}");
        }
    }

    #[test]
    fn a_keyword_that_is_also_a_built_in_is_offered_once() {
        let offered = labels("SELECT * FROM accounts WHERE email = NUL|");
        assert_eq!(
            offered.iter().filter(|label| *label == "NULL").count(),
            1,
            "{offered:?}"
        );
        let sql = "SELECT * FROM accounts WHERE email = NUL";
        let (_, null) = catalog()
            .items(sql, sql.len())
            .0
            .into_iter()
            .find(|(_, candidate)| candidate.label == "NULL")
            .expect("NULL was offered");
        assert_eq!(null.kind, CompletionItemKind::CONSTANT);
    }

    #[test]
    fn no_built_in_is_offered_where_a_relation_belongs_or_inside_a_literal() {
        assert!(!labels("SELECT * FROM date_tr|").contains(&"date_trunc".to_string()));
        assert!(
            !labels(r#"SELECT "o".date_tr| FROM orders o"#).contains(&"date_trunc".to_string())
        );
        assert!(labels("SELECT 'now|'").is_empty());
    }

    #[test]
    fn a_raw_filter_offers_built_ins() {
        let offered = filter_items(Engine::Postgres, "accounts", "created_at < no|");
        assert!(
            offered.contains(&("no".to_string(), "now".to_string())),
            "{offered:?}"
        );
    }

    #[test]
    fn the_list_is_capped() {
        let wide = (0..500)
            .map(|index| format!("column_{index}"))
            .collect::<Vec<_>>();
        let names = wide.iter().map(String::as_str).collect::<Vec<_>>();
        let completions = detached(
            Catalog {
                schemas: vec![Schema {
                    name: "public".to_string(),
                    relations: vec![relation("wide")],
                    routines: Vec::new(),
                    sequences: Vec::new(),
                }],
                database: None,
            },
            &[("public", "wide", &names)],
        );
        let sql = "SELECT wide. FROM wide";
        let offset = sql.find(' ').unwrap() + "wide.".len() + 1;
        assert_eq!(completions.items(sql, offset).0.len(), MAX_ITEMS);
    }

    #[test]
    fn a_relation_whose_columns_are_not_held_yet_is_asked_for_once() {
        let completions = detached(schemas(), &[]);
        let sql = "SELECT accounts.";

        let (items, wanted) = completions.items(sql, sql.len());
        assert!(items.is_empty(), "nothing is known about its columns yet");
        assert_eq!(wanted, vec![("public".to_string(), "accounts".to_string())]);

        // What the provider does on a miss, so the second keystroke asks for
        // nothing while the first fetch is still out.
        completions
            .columns
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(
                ("public".to_string(), "accounts".to_string()),
                ColumnState::Loading(0),
            );
        assert!(completions.items(sql, sql.len()).1.is_empty());
    }

    #[test]
    fn a_keystroke_asks_for_no_more_relations_than_may_be_in_flight() {
        let many = Catalog {
            schemas: (0..200)
                .map(|n| Schema {
                    name: format!("tenant_{n}"),
                    relations: vec![relation("users")],
                    routines: Vec::new(),
                    sequences: Vec::new(),
                })
                .collect(),
            database: None,
        };
        let completions = detached(many, &[]);
        let sql = "SELECT * FROM users WHERE u";

        let (_, wanted) = completions.items(sql, sql.len());
        assert_eq!(wanted.len(), 200, "every schema's users is wanted");
        let first = completions.admitted(wanted.clone());
        assert_eq!(first, wanted[..FETCHES_IN_FLIGHT]);

        completions
            .columns
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(first[0].clone(), ColumnState::Loading(0));
        assert_eq!(
            completions.admitted(wanted.clone()).len(),
            FETCHES_IN_FLIGHT - 1
        );
        for key in &first {
            completions
                .columns
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(key.clone(), ColumnState::Loading(0));
        }
        assert!(completions.admitted(wanted).is_empty());
    }

    #[test]
    fn a_name_the_catalog_does_not_list_is_never_asked_for() {
        // Or a typo would put a describe on the wire for every keystroke it
        // stays on screen.
        let completions = detached(schemas(), &[]);
        let sql = "SELECT accunts.";
        assert!(completions.items(sql, sql.len()).1.is_empty());
    }

    #[test]
    fn a_relation_the_server_would_not_describe_is_not_asked_again() {
        let completions = detached(schemas(), &[]);
        completions
            .columns
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(
                ("public".to_string(), "accounts".to_string()),
                ColumnState::Failed(FETCH_ATTEMPTS),
            );
        let sql = "SELECT accounts.";
        assert!(completions.items(sql, sql.len()).1.is_empty());
    }

    #[test]
    fn a_failure_is_retried_until_the_budget_runs_out() {
        // One blip -- or one statement timeout, which bounds dbdelve's own
        // catalog queries too -- must not silently kill completion for a
        // relation for the rest of the connection.
        let completions = detached(schemas(), &[]);
        let key = ("public".to_string(), "accounts".to_string());
        let sql = "SELECT accounts.";

        for attempts in 0..FETCH_ATTEMPTS {
            completions
                .columns
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(key.clone(), ColumnState::Failed(attempts));
            assert_eq!(
                completions.items(sql, sql.len()).1,
                vec![key.clone()],
                "a relation that has failed {attempts} times is still worth asking about"
            );
        }
    }

    #[test]
    fn only_the_relations_a_statement_names_are_asked_for() {
        // The whole point of the cache: writing about one table must not fetch
        // the columns of every table in the database.
        let completions = detached(schemas(), &[]);
        let sql = "SELECT e FROM audit.events";
        let (_, wanted) = completions.items(sql, sql.find(" FROM").unwrap());
        assert_eq!(wanted, vec![("audit".to_string(), "events".to_string())]);
    }

    /// The labels offered in a MongoDB buffer for a caret at `|`, over a
    /// catalog holding `users` (sampled: `_id`, `name`, `email`) and `orders`.
    fn mongo_labels(text: &str) -> Vec<String> {
        let offset = text.find('|').expect("the test must mark the caret");
        let text = text.replace('|', "");
        let mut completions = detached(
            Catalog {
                schemas: vec![Schema {
                    name: "shop".to_string(),
                    relations: vec![relation("users"), relation("orders")],
                    routines: Vec::new(),
                    sequences: Vec::new(),
                }],
                database: None,
            },
            &[("shop", "users", &["_id", "name", "email"])],
        );
        completions.engine = Engine::MongoDb;
        completions
            .items(&text, offset)
            .0
            .into_iter()
            .map(|(_, candidate)| candidate.label)
            .collect()
    }

    #[test]
    fn after_db_a_mongo_buffer_offers_collections_then_database_methods() {
        let offered = mongo_labels("db.|");
        assert_eq!(&offered[..2], ["users", "orders"]);
        assert!(offered.contains(&"runCommand".to_string()), "{offered:?}");
        assert_eq!(mongo_labels("db.ord|"), vec!["orders"]);
    }

    #[test]
    fn after_a_collection_its_methods_are_offered() {
        for text in [
            "db.users.|",
            "db.getCollection(\"users\").|",
            "db.getSiblingDB('other').users.|",
        ] {
            let offered = mongo_labels(text);
            assert!(
                offered.contains(&"insertOne".to_string()),
                "{text}: {offered:?}"
            );
            assert!(
                !offered.contains(&"limit".to_string()),
                "{text}: {offered:?}"
            );
        }
        assert_eq!(mongo_labels("db.users.countD|"), vec!["countDocuments"]);
    }

    #[test]
    fn after_find_the_cursor_methods_are_offered() {
        for text in [
            "db.users.find({ a: \")\" }).|",
            "db.users.find({}).sort({ name: 1 })\n  .|",
            "db.users.aggregate([]).|",
        ] {
            let offered = mongo_labels(text);
            assert!(
                offered.contains(&"limit".to_string()),
                "{text}: {offered:?}"
            );
            assert!(
                !offered.contains(&"insertOne".to_string()),
                "{text}: {offered:?}"
            );
        }
        // A cursor method is not offered after a method that returns none.
        assert!(mongo_labels("db.users.countDocuments({}).|").is_empty());
    }

    #[test]
    fn inside_a_document_fields_come_before_operators() {
        let offered = mongo_labels("db.users.find({ na|");
        assert_eq!(offered.first().map(String::as_str), Some("name"));
        assert!(mongo_labels("db.users.find({ age: { $g| } })").contains(&"$gte".to_string()));
        assert!(
            mongo_labels("db.getCollection('users').aggregate([{ $match: { em| } }])")
                .contains(&"email".to_string())
        );
        // No collection named, still operators.
        assert!(mongo_labels("db.runCommand({ $or| })").contains(&"$or".to_string()));
    }

    fn redis_labels(text: &str) -> Vec<String> {
        let offset = text.find('|').expect("the test must mark the caret");
        let text = text.replace('|', "");
        let mut completions = detached(
            Catalog {
                schemas: Vec::new(),
                database: None,
            },
            &[],
        );
        completions.engine = Engine::Redis;
        completions
            .items(&text, offset)
            .0
            .into_iter()
            .map(|(_, candidate)| candidate.label)
            .collect()
    }

    #[test]
    fn a_redis_buffer_offers_command_names_in_a_lines_first_word_only() {
        assert!(redis_labels("hge|").contains(&"HGETALL".to_string()));
        assert!(redis_labels("GET a\n  ft.cre|").contains(&"FT.CREATE".to_string()));
        assert!(redis_labels("GET ke|").is_empty());
        assert!(redis_labels("# ge|").is_empty());
        assert!(redis_labels("|").is_empty());
    }

    #[test]
    fn a_mongo_string_comment_or_regex_offers_nothing() {
        for text in [
            "db.users.find({ name: \"na|\" })",
            "db.users.find({ name: 'it\\'s na|' })",
            "// db.|",
            "db.users.find({ /* na| */ })",
            "db.users.find({ name: /na|/ })",
        ] {
            assert!(mongo_labels(text).is_empty(), "{text}");
        }
        assert!(!mongo_labels("db.users.find({ name: /x/ }).|").is_empty());
        assert!(!mongo_labels("// a note\ndb.|").is_empty());
    }

    /// What a raw filter on relation `name` offers for a caret at `|`: each label,
    /// with the text of the filter's own that accepting it would replace.
    ///
    /// Over a catalog whose relations are named the way a scan of SQL would
    /// misread, each with the same columns.
    fn filter_items(engine: Engine, name: &str, text: &str) -> Vec<(String, String)> {
        const AWKWARD: &[&str] = &["people's", "a--b", "has space", "say \"hi\"", "accounts"];
        let offset = text.find('|').expect("the test must mark the caret");
        let text = text.replace('|', "");
        let held = AWKWARD
            .iter()
            .map(|awkward| ("public", *awkward, &["id", "email"][..]))
            .collect::<Vec<_>>();
        let mut schema = detached(
            Catalog {
                schemas: vec![Schema {
                    name: "public".to_string(),
                    relations: AWKWARD.iter().map(|awkward| relation(awkward)).collect(),
                    routines: Vec::new(),
                    sequences: Vec::new(),
                }],
                database: None,
            },
            &held,
        );
        schema.engine = engine;
        let scope = ("public".to_string(), name.to_string());
        FilterCompletions::new(engine.syntax(), Arc::new(schema), scope)
            .expect("the engine completes a raw filter")
            .items(&text, offset)
            .0
            .into_iter()
            .map(|(word, candidate)| (text[word].to_string(), candidate.label))
            .collect()
    }

    #[test]
    fn a_raw_filter_offers_its_relations_columns_over_its_own_text() {
        for engine in [Engine::Postgres, Engine::MySql, Engine::SqlServer] {
            for relation in ["accounts", "people's", "a--b", "has space", "say \"hi\""] {
                let offered = filter_items(engine, relation, "id > 5 AND ema|");
                assert_eq!(
                    offered.first(),
                    Some(&("ema".to_string(), "email".to_string())),
                    "{engine:?} {relation}"
                );
            }
        }
        // At the very start of the text.
        let offered = filter_items(Engine::Postgres, "people's", "ema|");
        assert_eq!(
            offered.first(),
            Some(&("ema".to_string(), "email".to_string()))
        );
    }

    #[test]
    fn a_raw_filter_naming_its_own_relation_offers_and_asks_for_it_once() {
        let catalog = Catalog {
            schemas: vec![
                Schema {
                    name: "public".to_string(),
                    relations: vec![relation("accounts")],
                    routines: Vec::new(),
                    sequences: Vec::new(),
                },
                Schema {
                    name: "audit".to_string(),
                    relations: vec![relation("accounts")],
                    routines: Vec::new(),
                    sequences: Vec::new(),
                },
            ],
            database: None,
        };
        let text = "parent_id IN (SELECT id FROM accounts) AND ema";
        let scope = ("public".to_string(), "accounts".to_string());
        let filter = |held: &[(&str, &str, &[&str])]| {
            FilterCompletions::new(
                Syntax::Sql,
                Arc::new(detached(catalog.clone(), held)),
                scope.clone(),
            )
            .expect("SQL completes a raw filter")
            .items(text, text.len())
        };

        let (items, _) = filter(&[
            ("public", "accounts", &["id", "email"]),
            ("audit", "accounts", &["id", "embargo"]),
        ]);
        let labels: Vec<_> = items
            .iter()
            .map(|(_, candidate)| candidate.label.as_str())
            .collect();
        assert_eq!(
            labels.iter().filter(|label| **label == "email").count(),
            1,
            "{labels:?}"
        );
        // Another schema's relation of the same name is still the text's.
        assert!(labels.contains(&"embargo"), "{labels:?}");

        let (_, wanted) = filter(&[]);
        let asked = wanted.iter().filter(|key| **key == scope).count();
        assert_eq!(asked, 1, "{wanted:?}");
    }

    #[test]
    fn a_raw_filter_offers_nothing_inside_a_string() {
        assert!(filter_items(Engine::Postgres, "accounts", "email = 'ema|").is_empty());
    }

    #[test]
    fn a_mongo_raw_filter_offers_the_collections_fields() {
        for collection in ["accounts", "people's", "say \"hi\""] {
            let offered = filter_items(Engine::MongoDb, collection, "{ ema|");
            assert_eq!(
                offered.first(),
                Some(&("ema".to_string(), "email".to_string())),
                "{collection}"
            );
        }
        assert!(filter_items(Engine::MongoDb, "accounts", "{ email: \"ema|\" }").is_empty());
    }

    #[test]
    fn a_redis_raw_filter_completes_nothing() {
        let scope = ("0".to_string(), "keys".to_string());
        assert!(FilterCompletions::new(Syntax::Redis, Arc::new(catalog()), scope).is_none());
    }

    #[test]
    fn a_second_statement_does_not_borrow_the_first_statements_relations() {
        // Each statement is its own scope: the `accounts` in the first one must
        // not put `email` under the caret in the second.
        let offered = labels("SELECT * FROM accounts;\nSELECT ema| FROM audit.events");
        assert!(
            !offered.contains(&"email".to_string()),
            "a column leaked across a statement boundary: {offered:?}"
        );
    }

    fn key(column: &str, table: &str, referenced: &str) -> ForeignKey {
        ForeignKey {
            column: column.to_string(),
            referenced_schema: "public".to_string(),
            referenced_table: table.to_string(),
            referenced_column: referenced.to_string(),
        }
    }

    /// A shop whose every relation has been described, foreign keys and all,
    /// with aliasing on.
    fn shop() -> SchemaCompletions {
        let reference = |table: &str, column: &str| Reference {
            schema: "public".to_string(),
            table: table.to_string(),
            column: column.to_string(),
            referenced_column: "id".to_string(),
        };
        let described: Vec<(&str, Vec<ForeignKey>, Vec<Reference>)> = vec![
            (
                "customers",
                vec![],
                vec![reference("orders", "customer_id")],
            ),
            (
                "orders",
                vec![key("customer_id", "customers", "id")],
                vec![reference("order_items", "order_id")],
            ),
            ("order_items", vec![key("order_id", "orders", "id")], vec![]),
            (
                "employees",
                vec![key("manager_id", "employees", "id")],
                vec![reference("employees", "manager_id")],
            ),
            (
                "posts",
                vec![
                    key("created_by", "users", "id"),
                    key("updated_by", "users", "id"),
                ],
                vec![],
            ),
            ("users", vec![], vec![]),
            (
                "shipments",
                vec![
                    key("order_id", "order_lines", "order_id"),
                    key("line_no", "order_lines", "line_no"),
                ],
                vec![],
            ),
            ("order_lines", vec![], vec![]),
            ("OrderItems", vec![], vec![]),
            ("is_new", vec![], vec![]),
        ];
        let catalog = Catalog {
            schemas: vec![Schema {
                name: "public".to_string(),
                relations: described.iter().map(|(name, ..)| relation(name)).collect(),
                routines: Vec::new(),
                sequences: Vec::new(),
            }],
            database: None,
        };
        let held = described
            .into_iter()
            .map(|(name, foreign_keys, referenced_by)| {
                (
                    ("public".to_string(), name.to_string()),
                    ColumnState::Loaded(Described {
                        columns: vec!["id".to_string()],
                        foreign_keys,
                        referenced_by,
                    }),
                )
            })
            .collect();
        SchemaCompletions {
            columns: Arc::new(Mutex::new(held)),
            alias: true,
            ..detached(catalog, &[])
        }
    }

    /// What the popup shows and accepting writes, which are one text for an
    /// unqualified relation, for a caret at `|`.
    fn offered(completions: &SchemaCompletions, sql: &str) -> Vec<String> {
        let offset = sql.find('|').expect("the test must mark the caret");
        let sql = sql.replace('|', "");
        completions
            .items(&sql, offset)
            .0
            .into_iter()
            .map(|(_, candidate)| candidate.written(completions.engine))
            .collect()
    }

    fn first(sql: &str) -> String {
        offered(&shop(), sql).remove(0)
    }

    #[test]
    fn a_relation_after_from_is_written_with_its_initials() {
        assert_eq!(first("SELECT * FROM custo|"), "customers c");
        assert_eq!(first("SELECT * FROM order_it|"), "order_items oi");
        assert_eq!(first("SELECT * FROM OrderI|"), "OrderItems oi");
        assert_eq!(first("SELECT * FROM public.custo|"), "customers c");
    }

    #[test]
    fn an_alias_steps_around_the_statements_own_and_around_keywords() {
        assert_eq!(
            first("SELECT * FROM order_items oi JOIN OrderI|"),
            "OrderItems oi2"
        );
        assert_eq!(
            first("SELECT * FROM users o JOIN order_lin|"),
            "order_lines ol"
        );
        assert_eq!(first("SELECT * FROM users c JOIN custo|"), "customers c2");
        // `in` would not read back as a name.
        assert_eq!(first("SELECT * FROM is_n|"), "is_new in2");
        // The word being typed is not a relation the statement holds.
        assert_eq!(first("SELECT * FROM c|"), "customers c");
    }

    #[test]
    fn no_alias_is_written_where_none_belongs() {
        for sql in [
            "UPDATE custo|",
            "INSERT INTO custo|",
            "DELETE FROM custo|",
            "SELECT * FROM custo| c",
            "SELECT * FROM custo| AS c",
            "SELECT * FROM custo|mers",
        ] {
            assert_eq!(first(sql), "customers", "{sql}");
        }
        assert_eq!(first("SELECT * FROM custo| WHERE true"), "customers c");
        assert_eq!(
            first("SELECT * FROM (SELECT * FROM custo|) x"),
            "customers c"
        );

        let off = SchemaCompletions {
            alias: false,
            ..shop()
        };
        assert_eq!(
            offered(&off, "SELECT * FROM orders o JOIN custo|")[0],
            "customers"
        );
    }

    #[test]
    fn a_join_is_offered_with_the_condition_a_foreign_key_implies() {
        // The key is the joined relation's own, found from who points back.
        assert_eq!(
            offered(&shop(), "SELECT * FROM orders o JOIN order_it|")[..2],
            ["order_items oi ON oi.order_id = o.id", "order_items oi"]
        );
        // The key is the held relation's.
        assert_eq!(
            first("SELECT * FROM orders o JOIN cust|"),
            "customers c ON c.id = o.customer_id"
        );
        // A relation with no alias is named in full.
        assert_eq!(
            first("SELECT * FROM public.orders LEFT JOIN cust|"),
            "customers c ON c.id = orders.customer_id"
        );
        assert_eq!(
            first("SELECT * FROM orders o JOIN public.cust|"),
            "customers c ON c.id = o.customer_id"
        );
    }

    #[test]
    fn a_join_leads_a_relation_that_matches_better() {
        assert_eq!(
            offered(&shop(), "SELECT * FROM order_items oi JOIN o|")[0],
            "orders o ON o.id = oi.order_id"
        );
    }

    #[test]
    fn keys_onto_one_relation_are_told_apart_from_one_key_of_many_columns() {
        assert_eq!(
            offered(&shop(), "SELECT * FROM posts p JOIN us|")[..3],
            [
                "users u ON u.id = p.created_by",
                "users u ON u.id = p.updated_by",
                "users u"
            ]
        );
        assert_eq!(
            first("SELECT * FROM shipments s JOIN order_l|"),
            "order_lines ol ON ol.order_id = s.order_id AND ol.line_no = s.line_no"
        );
    }

    #[test]
    fn a_relation_already_held_is_joined_again_only_to_itself() {
        assert_eq!(
            offered(&shop(), "SELECT * FROM employees e JOIN emp|"),
            [
                "employees e2 ON e2.id = e.manager_id",
                "employees e2 ON e2.manager_id = e.id",
                "employees e2"
            ]
        );
        assert_eq!(
            offered(
                &shop(),
                "SELECT * FROM orders o JOIN order_items oi ON oi.order_id = o.id JOIN ord|"
            )[0],
            "orders o2"
        );
    }

    #[test]
    fn no_condition_is_written_in_front_of_the_statements_own() {
        assert_eq!(
            first("SELECT * FROM orders o JOIN cust| ON true"),
            "customers c"
        );
        assert_eq!(first("SELECT * FROM orders o, cust|"), "customers");
    }

    #[test]
    fn no_condition_follows_a_join_that_takes_none_or_names_a_relation_out_of_reach() {
        for sql in [
            "SELECT * FROM orders o CROSS JOIN cust|",
            "SELECT * FROM orders o NATURAL LEFT JOIN cust|",
            "SELECT * FROM orders o JOIN cust| /* keep */ ON true",
            "WITH x AS (SELECT * FROM orders o) SELECT * FROM users u JOIN cust|",
        ] {
            assert!(!first(sql).contains(" ON "), "{sql}");
        }
        assert_eq!(
            first("SELECT * FROM users u JOIN (SELECT * FROM orders o JOIN cust|"),
            "customers c ON c.id = o.customer_id"
        );
    }

    #[test]
    fn a_condition_names_only_what_the_query_at_the_caret_can_see() {
        for sql in [
            "SELECT 'FROM orders o' AS note FROM users u JOIN cust|",
            "SELECT 1 /* FROM orders o */ FROM users u JOIN cust|",
            "WITH orders AS (SELECT id FROM users) SELECT * FROM orders o JOIN cust|",
            "WITH orders AS MATERIALIZED (SELECT 1) SELECT * FROM orders o JOIN cust|",
            "SELECT o.id FROM orders o UNION ALL SELECT u.id FROM users u JOIN cust|",
        ] {
            assert_eq!(first(sql), "customers c", "{sql}");
        }
        assert_eq!(
            first("SELECT * FROM orders /* the shop's */ o JOIN cust|"),
            "customers c ON c.id = o.customer_id"
        );
    }

    #[test]
    fn a_relation_the_statement_reaches_into_by_name_keeps_its_name() {
        assert_eq!(first("SELECT customers.id FROM custo|"), "customers");
        assert_eq!(
            first("SELECT customers.id FROM orders o JOIN custo|"),
            "customers ON customers.id = o.customer_id"
        );
        assert_eq!(first("SELECT o.id FROM custo|"), "customers c");
        // One in a subquery is not beside it.
        assert_eq!(
            first("SELECT employees.id FROM emp| WHERE EXISTS (SELECT 1 FROM employees e)"),
            "employees"
        );
        assert_eq!(
            first("SELECT employees.id FROM employees JOIN emp|"),
            "employees e ON e.id = employees.manager_id"
        );
    }

    #[test]
    fn a_key_into_a_schema_the_catalog_does_not_hold_joins_nothing() {
        let completions = shop();
        assert_eq!(
            completions.listed("public", "users"),
            Some(("public".to_string(), "users".to_string()))
        );
        assert_eq!(completions.listed("elsewhere", "users"), None);
        assert_eq!(
            completions.listed("", "users"),
            Some(("public".to_string(), "users".to_string()))
        );
    }

    #[test]
    fn two_keys_of_two_columns_onto_one_relation_stay_whole() {
        let keys = [
            key("billing_tenant", "addresses", "tenant_id"),
            key("billing_id", "addresses", "id"),
            key("shipping_tenant", "addresses", "tenant_id"),
            key("shipping_id", "addresses", "id"),
        ];
        let columns = foreign_keys(&keys)
            .into_iter()
            .map(|(_, columns)| columns.len())
            .collect::<Vec<_>>();
        assert_eq!(columns, [2, 2]);
    }

    #[test]
    fn a_join_asks_for_the_keys_of_a_relation_not_yet_described() {
        let completions = SchemaCompletions {
            columns: ColumnCache::default(),
            ..shop()
        };
        let sql = "SELECT * FROM orders o JOIN cust";
        let (items, wanted) = completions.items(sql, sql.len());
        assert_eq!(wanted, [("public".to_string(), "orders".to_string())]);
        assert_eq!(items[0].1.written(Engine::Postgres), "customers c");
    }

    #[test]
    fn a_filter_bar_writes_no_alias() {
        let scope = ("public".to_string(), "orders".to_string());
        let bar = FilterCompletions::new(Syntax::Sql, Arc::new(shop()), scope).unwrap();
        let text = "id IN (SELECT id FROM cust";
        let (items, _) = bar.items(text, text.len());
        assert_eq!(items[0].1.written(Engine::Postgres), "customers");
    }

    /// A window with nothing in it, for the one test that must hand the
    /// provider a real `Window` without dragging the workspace up.
    struct Bare;

    impl gpui::Render for Bare {
        fn render(
            &mut self,
            _: &mut gpui::Window,
            _: &mut gpui::Context<Self>,
        ) -> impl gpui::IntoElement {
            gpui::div()
        }
    }

    #[gpui::test]
    fn a_request_runs_its_work_off_the_calling_thread(cx: &mut gpui::TestAppContext) {
        use std::{cell::RefCell, rc::Rc};

        use lsp_types::CompletionTriggerKind;

        let completions = Arc::new(catalog());
        let sql = "SELECT * FROM acc";
        let rope = Rope::from(sql);

        let (_, visual) = cx.add_window_view(|_, _| Bare);
        let response = visual.update(|window, cx| {
            let request = completions.completions(
                &rope,
                sql.len(),
                CompletionContext {
                    trigger_kind: CompletionTriggerKind::INVOKED,
                    trigger_character: None,
                },
                window,
                cx,
            );
            // The call has returned and the parse has not seen the buffer it
            // is about to rank from. The test scheduler runs background tasks
            // only on a drain, so nothing of the compute can have started
            // yet: this pins the work sitting behind a spawn rather than on
            // the calling thread, where the keystroke would wait for it.
            assert!(
                completions
                    .parsed
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .held()
                    .is_empty()
            );
            // Held past this scope, since dropping the request would cancel
            // the compute behind it; a witness on the foreground executor
            // awaits the response the way the editor does.
            let seen = Rc::new(RefCell::new(None));
            let witness = seen.clone();
            cx.spawn(async move |_| {
                *witness.borrow_mut() = request.await.ok();
            })
            .detach();
            seen
        });

        cx.run_until_parked();

        // The compute ran, off the calling thread, and delivered the response
        // through it -- the same leading row the direct decision (`items`)
        // gives.
        assert_eq!(
            completions
                .parsed
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .held()
                .as_ref(),
            sql
        );
        let response = response.borrow();
        let response = response.as_ref().expect("the response landed");
        let CompletionResponse::Array(items) = response else {
            unreachable!("the provider answers with an array");
        };
        assert_eq!(
            items.first().expect("accounts was offered").label,
            "accounts"
        );
    }
}
