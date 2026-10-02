"""Retired direct E5 profile updater; active generations must remain immutable."""


def main() -> None:
    raise SystemExit(
        "Direct updates to category_search_profiles_v3 are disabled. "
        "Use `python -m model_intent embeddings sync` with the pinned E5 configuration "
        "to build or reuse a versioned generation in the unified database."
    )


if __name__ == "__main__":
    main()
