"""Resolve held-out validation candidates from the training protocol.

The selected development file is separate from the final test candidates.
Changing the development objective requires a new experiment identity.
"""
from pathlib import Path

DEV_PROTOCOL_FILES = {"imp": "behaviors.tsv", "rs50": "behaviors_rs50.tsv"}


def resolve_dev_protocol(train_type, requested="auto"):
    """Return the matching dev protocol, rejecting a conflicting override."""
    if train_type not in DEV_PROTOCOL_FILES:
        raise ValueError("Unsupported training protocol: " + str(train_type))
    if requested in (None, "auto"):
        return train_type
    if requested != train_type:
        raise ValueError(
            "Training protocol %s requires matching %s development, not %s"
            % (train_type, train_type, requested)
        )
    return train_type


def resolve_dev_path(data_root, train_type, requested="auto"):
    """Select the matching held-out dev file, never a test file."""
    protocol = resolve_dev_protocol(train_type, requested)
    return (Path(data_root).resolve() / "dev" / DEV_PROTOCOL_FILES[protocol]).resolve()



def validate_candidate_profile(data_root, train_type):
    """Require the verified, request-time-only RandS train/dev preparation."""
    resolve_dev_protocol(train_type)
    root = Path(data_root).resolve()
    if (root / "consistent_candidate_manifest.json").is_file():
        from consistent_candidate_profile import validate_consistent_candidate_profile
        return validate_consistent_candidate_profile(root)
    if train_type == "imp":
        return None
    from prepare_development_candidates import validate_matched_candidate_profile
    return validate_matched_candidate_profile(data_root)
