"""Design cells that the statistical tests can be restricted to.

The study is 391 participants x 3 topics = 1173 observations, and within each
topic a participant was shown one of three comment packages. The full design is
therefore 9 topic x package cells of roughly 130 observations each:

    topic       UBI, weight_loss, penalty
    package     package1, package2, package3   (nested inside a topic)

Every test in ``scripts/stats/`` is normally run on all 1173 observations. A
:class:`Group` restricts one to a single cell. ``ALL`` -- both fields ``None``
-- is the unrestricted group and is the default argument everywhere, so the
ungrouped numbers keep coming out of exactly the same code path as before.

Two properties of the design matter when reading subset results:

* A participant appears **once** in any single cell (one observation per topic,
  and one package per participant-topic). The within-participant clustering
  that separates the nominal n from the independent n therefore disappears
  inside a cell -- there, the two sample sizes coincide.
* The 27 comments are 3 topics x 3 packages x 3 messages, so a topic cell holds
  9 comments and a topic x package cell holds only 3. Tests that operate on the
  comment means (``comment_rank_variability``) run on groups of that size.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config import MERGED_HUMAN_CSV

TOPICS = ("UBI", "weight_loss", "penalty")
PACKAGES = ("package1", "package2", "package3")


@dataclass(frozen=True)
class Group:
    """A subset of the design: a topic, a topic x package cell, or everything."""

    topic: Optional[str] = None
    package: Optional[str] = None

    @property
    def is_all(self):
        return self.topic is None and self.package is None

    @property
    def label(self):
        """Filename-safe identifier."""
        if self.topic is None:
            return "overall"
        if self.package is None:
            return self.topic
        return f"{self.topic}__{self.package}"

    @property
    def title(self):
        """Human-readable description for report headers."""
        if self.topic is None:
            return "all topics and packages"
        if self.package is None:
            return f"topic = {self.topic}"
        return f"topic = {self.topic}, package = {self.package}"

    def matches(self, topic, package=None):
        """Does a single record fall in this group?

        `package` may be left out when the group does not constrain it; a group
        that does constrain it needs the value and rejects a missing one.
        """
        if self.topic is not None and topic != self.topic:
            return False
        if self.package is not None and package != self.package:
            return False
        return True

    def filter(self, df, topic_column="topic", package_column="package"):
        """The rows of `df` in this group, as a new frame.

        An empty result is an error rather than an empty table: every cell of
        this design is populated, so emptiness means the column names or the
        topic/package spellings are wrong, and silently testing zero rows would
        surface much later as a confusing traceback inside scipy.
        """
        if self.is_all:
            return df
        mask = pd.Series(True, index=df.index)
        if self.topic is not None:
            mask &= df[topic_column] == self.topic
        if self.package is not None:
            mask &= df[package_column] == self.package
        out = df[mask]
        if out.empty:
            raise ValueError(
                f"no rows for {self.title}; columns carry "
                f"topic={sorted(df[topic_column].dropna().unique())[:5]} "
                f"package={sorted(df[package_column].dropna().unique())[:5]}"
            )
        return out


ALL = Group()

#: CLI name -> the groups it runs. Keys are the values of ``--group-by``.
GROUPINGS = {
    "overall": (ALL,),
    "topic": tuple(Group(topic=t) for t in TOPICS),
    "topic-package": tuple(Group(topic=t, package=p)
                           for t in TOPICS for p in PACKAGES),
}


def design_counts(group=ALL):
    """``{n_obs, n_participants}`` for a group, read from the human data.

    Inside a cell each participant contributes exactly one observation, so the
    two counts are equal there; over the whole design they are 1173 and 391.
    """
    df = group.filter(pd.read_csv(MERGED_HUMAN_CSV))
    return {"n_obs": int(len(df)),
            "n_participants": int(df["persona_id"].nunique())}
