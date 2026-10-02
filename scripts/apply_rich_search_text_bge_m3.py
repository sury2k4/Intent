"""Retired direct BGE profile updater; the serving model is E5-base 768D."""


def main() -> None:
    raise SystemExit(
        "Direct updates to category_search_profiles_bge_m3 are disabled. "
        "The unified serving database uses pinned E5-base 768D; "
        "BGE-M3 has no supported versioned sync path in this project."
    )


if __name__ == "__main__":
    main()
