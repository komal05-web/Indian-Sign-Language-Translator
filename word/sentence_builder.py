"""
sentence_builder.py – ISL Letter-to-Sentence Engine
=====================================================
Converts frame-by-frame letter detections into words and full sentences,
and matches completed sentences against a built-in ISL phrase dictionary.

Controls (handled in predict.py):
    SPACE      → confirm current word, start next word
    BACKSPACE  → delete last letter (or restore last word)
    ENTER      → speak / finalise the full sentence
    C          → clear everything
"""


# ── Basic ISL phrase dictionary ────────────────────────────────────────────────
# Keys   : the word / sentence spelled out in CAPITAL letters (spaces removed).
# Values : the natural language expansion shown / spoken.
#
# Dataset used for phrase vocabulary reference:
#   INCLUDE – A Large Scale Dataset for Indian Sign Language Recognition
#   https://zenodo.org/records/4010759
#
# Feel free to extend this dict with your own domain phrases.

BASIC_ISL_PHRASES: dict[str, str] = {
    # ── Greetings ──────────────────────────────────────────────────────────
    "HELLO":        "Hello!",
    "HI":           "Hi there!",
    "BYE":          "Goodbye!",
    "GOODMORNING":  "Good morning!",
    "GOODNIGHT":    "Good night!",
    "WELCOME":      "You are welcome!",

    # ── Courtesy ───────────────────────────────────────────────────────────
    "THANKS":       "Thank you!",
    "THANKYOU":     "Thank you very much!",
    "SORRY":        "I am sorry!",
    "PLEASE":       "Please.",
    "EXCUSE":       "Excuse me.",

    # ── Yes / No ───────────────────────────────────────────────────────────
    "YES":          "Yes.",
    "NO":           "No.",
    "OK":           "Okay.",
    "FINE":         "I am fine.",

    # ── Common questions ───────────────────────────────────────────────────
    "HOWRU":        "How are you?",
    "WHATSYOURNAME":"What is your name?",
    "NAME":         "What is your name?",
    "WHERE":        "Where is it?",
    "WHEN":         "When?",
    "WHY":          "Why?",
    "WHO":          "Who?",
    "WHAT":         "What?",

    # ── Needs ─────────────────────────────────────────────────────────────
    "HELP":         "I need help!",
    "WATER":        "I need water.",
    "FOOD":         "I want food.",
    "EAT":          "I want to eat.",
    "DRINK":        "I want to drink.",
    "SLEEP":        "I want to sleep.",
    "TOILET":       "I need the toilet.",
    "MEDICINE":     "I need medicine.",
    "DOCTOR":       "I need a doctor.",
    "HOSPITAL":     "Take me to the hospital.",

    # ── Commands ───────────────────────────────────────────────────────────
    "STOP":         "Please stop.",
    "COME":         "Please come here.",
    "GO":           "Please go.",
    "WAIT":         "Please wait.",
    "SIT":          "Please sit down.",
    "STAND":        "Please stand up.",
    "OPEN":         "Please open the door.",
    "CLOSE":        "Please close the door.",

    # ── Emotions ───────────────────────────────────────────────────────────
    "LOVE":         "I love you.",
    "HAPPY":        "I am happy.",
    "SAD":          "I am sad.",
    "ANGRY":        "I am angry.",
    "TIRED":        "I am tired.",
    "SCARED":       "I am scared.",
    "PAIN":         "I am in pain.",
    "SICK":         "I am sick.",

    # ── Polite chat ────────────────────────────────────────────────────────
    "GOOD":         "Good.",
    "BAD":          "Bad.",
    "NICE":         "That is nice.",
    "UNDERSTAND":   "I understand.",
    "DONTUNDERSTAND":"I do not understand.",
    "REPEAT":       "Please repeat that.",
    "SLOW":         "Please speak slowly.",
    "MYNAME":       "My name is…",
}


class SentenceBuilder:
    """
    Accumulates per-frame letter detections into words and sentences.

    Letter-commit logic
    -------------------
    A letter must remain *stable* (identical) for ``hold_frames`` consecutive
    frames before it is committed to the word buffer.  After a commit a
    ``cooldown_frames``-long gap prevents the same letter being committed again
    immediately.

    Parameters
    ----------
    hold_frames : int
        Number of consecutive matching frames required to commit a letter.
        Lower → more responsive; higher → fewer accidental presses.
        Recommended: 15–20 at 30 fps.
    cooldown_frames : int
        Frames to ignore input after each commit.
        Recommended: 20–30 at 30 fps.
    """

    def __init__(self, hold_frames: int = 18, cooldown_frames: int = 25):
        self.hold_frames     = hold_frames
        self.cooldown_frames = cooldown_frames

        self.current_word: list[str] = []   # letters being built right now
        self.sentence:     list[str] = []   # finalised words

        self._last_letter:   str | None = None
        self._hold_count:    int        = 0
        self._cooldown_left: int        = 0

    # ── Core letter feed ──────────────────────────────────────────────────────

    def feed(self, letter: str) -> bool:
        """
        Pass the latest detected letter each frame.

        Returns
        -------
        bool
            True exactly on the frame when a new letter is committed to the
            word buffer (useful for audio feedback / UI flash).
        """
        if not letter or letter in ("-", "?", ""):
            self._reset_hold()
            return False

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return False

        if letter == self._last_letter:
            self._hold_count += 1
        else:
            self._last_letter = letter
            self._hold_count  = 1
            return False

        if self._hold_count >= self.hold_frames:
            self.current_word.append(letter)
            self._reset_hold()
            self._cooldown_left = self.cooldown_frames
            return True          # ← letter just committed

        return False

    def _reset_hold(self) -> None:
        self._last_letter = None
        self._hold_count  = 0

    # ── Manual controls ───────────────────────────────────────────────────────

    def space(self) -> None:
        """Push current word into the sentence and start a new word."""
        word = self.word_str.strip()
        if word:
            self.sentence.append(word)
        self.current_word.clear()
        self._reset_hold()

    def backspace(self) -> None:
        """
        Delete the last committed character.
        If the word buffer is empty, restore the last completed word for editing.
        """
        if self.current_word:
            self.current_word.pop()
        elif self.sentence:
            self.current_word = list(self.sentence.pop())
        self._reset_hold()

    def clear(self) -> None:
        """Wipe the entire sentence and word buffers."""
        self.current_word.clear()
        self.sentence.clear()
        self._reset_hold()
        self._cooldown_left = 0

    # ── Read-only properties ──────────────────────────────────────────────────

    @property
    def word_str(self) -> str:
        """Current word being assembled."""
        return "".join(self.current_word)

    @property
    def sentence_str(self) -> str:
        """Full sentence including the in-progress word."""
        parts = self.sentence[:]
        if self.current_word:
            parts.append(self.word_str)
        return " ".join(parts)

    @property
    def hold_progress(self) -> float:
        """
        Fractional progress (0.0 – 1.0) of the current letter hold.
        Use this to draw a visual progress ring / bar.
        """
        if self._hold_count == 0 or self.hold_frames == 0:
            return 0.0
        return min(self._hold_count / self.hold_frames, 1.0)

    @property
    def is_empty(self) -> bool:
        return not self.current_word and not self.sentence

    # ── Phrase matching ───────────────────────────────────────────────────────

    def matched_phrase(self) -> str:
        """
        Check whether the typed sentence matches a known ISL phrase.

        Matching is done by collapsing all spaces and uppercasing, so
        'GOOD MORNING' and 'GOODMORNING' both hit the same key.

        Returns the expanded phrase string, or '' if no match.
        """
        key = self.sentence_str.replace(" ", "").upper()
        return BASIC_ISL_PHRASES.get(key, "")

    def all_phrases(self) -> dict[str, str]:
        """Return the full phrase dictionary (for UI cheat-sheet display)."""
        return dict(BASIC_ISL_PHRASES)
