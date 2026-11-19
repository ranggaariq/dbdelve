//! The keycast badge: a brief flash above the status bar naming the shortcut
//! just fired -- its chord, from the registry so a rebind is what shows, and
//! the action's own label.
//!
//! It is pure drawing: no focus, no hitbox, no listener, so a keystroke can
//! never land on it and a click passes through it. Its state is the flash on
//! `Workspace`, never element state, because a flash that replaces another
//! must replay its fade rather than continue someone else's.

use std::time::Duration;

use gpui::{Animation, AnimationExt, Keystroke};

use super::*;

/// How long the fade-in runs. It plays once and never repeats, so it is not
/// the redraw hazard `AGENTS.md` warns about; `reduce_motion` renders the
/// one-shot animation's end state instead of scheduling frames.
const FADE: Duration = Duration::from_millis(150);

/// How long a flash stays up before the badge clears itself.
const HOLD: Duration = Duration::from_secs(1);

/// One flash of the badge.
pub(crate) struct Flash {
    /// Which flash this is: what the element is keyed by, so a new chord
    /// remounts and replays the fade, and what the dismissal timer carries,
    /// so a replaced chord's timer cannot clear what replaced it.
    pub(crate) seq: u64,
    /// The chord, as the registry spells it -- `secondary-k r` for both
    /// strokes, not the `r` that completed it.
    pub(crate) chord: String,
    pub(crate) label: &'static str,
}

/// Whether `chord` is one worth flashing: only a chord carrying a modifier
/// beyond shift. Palette and dialog navigation (`up`, `shift-tab`, `enter`)
/// is the daily bread of the keyboard here -- flashing each stroke of it
/// would be noise, not news.
pub(crate) fn flashes(chord: &str) -> bool {
    chord.split_whitespace().any(|stroke| {
        let Ok(stroke) = Keystroke::parse(stroke) else {
            return false;
        };
        let held = stroke.modifiers;
        held.control || held.alt || held.platform || held.function
    })
}

impl Workspace {
    /// Puts `chord` and `label` up on the badge, replacing whatever is
    /// showing, and arms the one timer that takes it down. The timer checks
    /// the flash it armed for rather than clearing blindly: this chord may
    /// already have been replaced by the time it fires.
    pub(crate) fn flash_keycast(
        &mut self,
        label: &'static str,
        chord: &str,
        cx: &mut Context<Self>,
    ) {
        self.keycast_seq += 1;
        self.keycast = Some(Flash {
            seq: self.keycast_seq,
            chord: chord.to_string(),
            label,
        });
        cx.notify();
        let seq = self.keycast_seq;
        // ponytail: one detached task per flash, so rapid repeat holds several
        // stale timers for under a second each, all no-ops through the seq
        // guard. If that ever shows, hold the previous Task in a field and
        // cancel it on replace.
        cx.spawn(async move |workspace, cx| {
            cx.background_executor().timer(HOLD).await;
            workspace
                .update(cx, |workspace, cx| {
                    if workspace
                        .keycast
                        .as_ref()
                        .is_some_and(|flash| flash.seq == seq)
                    {
                        workspace.keycast = None;
                        cx.notify();
                    }
                })
                .ok();
        })
        .detach();
    }
}

/// The badge itself: the chord as keycaps, the label under it, hung
/// bottom-centre at the status bar's height and fading in once.
pub(crate) fn layer(flash: &Flash, t: Theme) -> impl IntoElement + use<> {
    // A full-width row to centre in: the badge itself carries no listener
    // and no hitbox, so the row lets every click through to what is under
    // it -- it is a place to stand, not a surface.
    div()
        .absolute()
        .left_0()
        .right_0()
        .bottom(px(
            layout::chrome(layout::STATUS_HEIGHT) + layout::chrome(layout::SPACE_MD)
        ))
        .flex()
        .justify_center()
        .child(
            div()
                .debug_selector(|| "keycast-badge".into())
                .flex()
                .flex_col()
                .items_center()
                .gap(px(layout::SPACE_SM))
                .px(px(layout::SPACE_MD))
                .py(px(layout::SPACE_SM))
                .border_1()
                .border_color(t.border)
                .rounded(px(layout::RADIUS_PANEL))
                .map(|card| ui::raised_card(card, t))
                // Keyed by the flash, so a new chord remounts and the fade
                // plays from the top. Under `reduce_motion` the framework
                // draws the one-shot animation's end state and schedules no
                // frames: the fade is what is skipped, never the badge.
                .with_animation(
                    ("keycast", flash.seq),
                    Animation::new(FADE),
                    |card, delta| card.opacity(delta),
                )
                .child(ui::keycaps(&flash.chord, t))
                .child(
                    div()
                        .text_size(px(layout::chrome(layout::TEXT_XS)))
                        .text_color(t.text_muted)
                        .child(flash.label),
                ),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chord_with_a_modifier_beyond_shift_flashes() {
        for chord in ["alt-down", "secondary-k r", "ctrl-tab", "cmd-m"] {
            assert!(flashes(chord), "{chord}");
        }
    }

    #[test]
    fn plain_and_shift_only_chords_stay_quiet() {
        // The palette's and the dialogs' navigation, which fires registry
        // actions on nearly every keystroke while they are open.
        for chord in ["down", "up", "enter", "tab", "shift-tab", "space"] {
            assert!(!flashes(chord), "{chord}");
        }
    }

    #[test]
    fn one_modified_stroke_is_enough() {
        // A mixed chord flashes; a stroke nothing could parse counts as
        // nothing rather than taking the whole chord with it.
        assert!(flashes("shift-x ctrl-y"));
        assert!(!flashes("shift-x not-a-key"));
    }
}
