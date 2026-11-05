"""The label space: configs/class_mapping.yaml and configs/attributes.yaml, resolved.

Nothing downstream should parse either config itself. Datasets, models and eval all go
through LabelSpace and AttributeSpace so that there is exactly one place where a class
name becomes an index, and exactly one place that decides what is excluded.

Every lookup failure raises MappingError. There is deliberately no fallback, no "unknown"
bucket and no silent drop: CLAUDE.md section 4 makes an unmapped class a hard error,
because a class quietly vanishing from a 5-part differential is the kind of bug that
produces a plausible-looking number and a wrong conclusion.
"""
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
MAPPING_PATH = CONFIGS / "class_mapping.yaml"
ATTRIBUTES_PATH = CONFIGS / "attributes.yaml"

# Keys in class_mapping.yaml that are not target classes. Anything else at the top level
# is taken to be one of the five differential classes, so a new section must be added
# here -- the 5-class check fails loudly otherwise, which is the intended behaviour.
RESERVED = ("excluded", "excluded_files", "secondary_subsets", "target_splits")
N_CLASSES = 5


class MappingError(Exception):
    """A class name, attribute value or exclusion could not be resolved."""


class LabelSpace:
    """Maps "<dataset>:<class name>" onto a contiguous class index, and holds exclusions.

    Index order is the key order of class_mapping.yaml, which is the 5-part differential
    order the project reports in. Changing that order invalidates every checkpoint.
    """

    def __init__(self, mapping):
        self.classes = [key for key in mapping if key not in RESERVED]
        if len(self.classes) != N_CLASSES:
            raise MappingError(
                f"expected {N_CLASSES} target classes (the 5-part differential), "
                f"found {len(self.classes)}: {self.classes}"
            )

        self._index = {}
        for index, cls in enumerate(self.classes):
            for name in mapping[cls]:
                if name in self._index:
                    raise MappingError(
                        f"{name!r} is listed under more than one class: "
                        f"{self.classes[self._index[name]]!r} and {cls!r}"
                    )
                self._index[name] = index

        self._excluded = dict(mapping.get("excluded") or {})
        both = sorted(set(self._index) & set(self._excluded))
        if both:
            raise MappingError(f"listed as both mapped and excluded: {', '.join(both)}")

        self._files, self._folders = {}, {}
        for key, reason in (mapping.get("excluded_files") or {}).items():
            dataset, rel = key.split(":", 1)
            target = self._folders if rel.endswith("/") else self._files
            target.setdefault(dataset, {})[rel.rstrip("/")] = reason

        self.secondary_subsets = dict(mapping.get("secondary_subsets") or {})

    # ------------------------------------------------------------------ lookups

    def index_for(self, name):
        """Class index for "<dataset>:<class name>". Raises on anything else."""
        try:
            return self._index[name]
        except KeyError:
            if name in self._excluded:
                raise MappingError(
                    f"{name!r} is explicitly excluded ({self._excluded[name]}); "
                    f"it has no class index"
                ) from None
            raise MappingError(
                f"{name!r} is not mapped to any of {self.classes} and is not listed under "
                f"`excluded` in {MAPPING_PATH.name}. Add it to one or the other."
            ) from None

    def validate_observed(self, names):
        """Raise unless every observed class name is mapped or explicitly excluded."""
        unknown = sorted(n for n in names if n not in self._index and n not in self._excluded)
        if unknown:
            raise MappingError(
                f"{len(unknown)} class name(s) found in the data but absent from "
                f"{MAPPING_PATH.name}: {', '.join(unknown)}"
            )

    def is_excluded(self, name):
        return name in self._excluded

    # --------------------------------------------------------------- exclusions

    def excluded_files(self, dataset):
        """Dataset-relative paths of individually excluded files."""
        return set(self._files.get(dataset, {}))

    def excluded_folders(self, dataset):
        """Dataset-relative folder names excluded wholesale."""
        return set(self._folders.get(dataset, {}))

    def excluded_image_names(self):
        """Bare filenames of every excluded file, across datasets.

        WBCAtt identifies images by filename alone, and filenames are unique across all
        of PBC, so matching on the basename is enough to drop a row.
        """
        return {Path(rel).name for files in self._files.values() for rel in files}

    def exclusion_reason(self, dataset, rel):
        return self._files.get(dataset, {}).get(rel) or self._folders.get(dataset, {}).get(rel)


class AttributeSpace:
    """The 11 morphological attributes, one softmax head each.

    Vocabularies come from configs/attributes.yaml rather than from the data, so a value
    appearing or disappearing changes a config (and fails loudly) instead of silently
    changing a head's width.
    """

    def __init__(self, config):
        vocabularies = config["attributes"]
        self.heads = list(vocabularies)
        self._values = {}
        for head, values in vocabularies.items():
            # Bare no/yes/on/off are booleans in YAML, so an unquoted vocabulary silently
            # stops matching the CSV's strings. Catch it here rather than at lookup time.
            non_strings = [v for v in values if not isinstance(v, str)]
            if non_strings:
                raise MappingError(
                    f"attribute {head!r} has non-string value(s) {non_strings} in "
                    f"{ATTRIBUTES_PATH.name}; quote them (YAML reads bare no/yes as booleans)"
                )
            if len(set(values)) != len(values):
                duplicated = sorted({v for v in values if values.count(v) > 1})
                raise MappingError(
                    f"attribute {head!r} lists duplicate value(s): {', '.join(duplicated)}"
                )
            self._values[head] = {value: index for index, value in enumerate(values)}
        self.head_sizes = [len(self._values[head]) for head in self.heads]
        self.known_inconsistent_rows = set(config.get("known_inconsistent_rows") or ())

    def index_for(self, head, value):
        try:
            vocabulary = self._values[head]
        except KeyError:
            raise MappingError(
                f"{head!r} is not one of the {len(self.heads)} attributes in "
                f"{ATTRIBUTES_PATH.name}: {', '.join(self.heads)}"
            ) from None
        try:
            return vocabulary[value]
        except KeyError:
            raise MappingError(
                f"attribute {head!r} has no value {value!r} in {ATTRIBUTES_PATH.name}; "
                f"known values are {', '.join(vocabulary)}"
            ) from None

    def vector_for(self, row):
        """One index per head, in head order, from a WBCAtt CSV row."""
        return tuple(self.index_for(head, row[head]) for head in self.heads)


def load_label_space(path=MAPPING_PATH):
    return LabelSpace(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_attribute_space(path=ATTRIBUTES_PATH):
    return AttributeSpace(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
