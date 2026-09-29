from rag.tokenizer import tokenize


def test_lowercase():
    assert tokenize("FastAPI Is Great") == ["fastapi", "is", "great"]


def test_punctuation_is_removed():
    assert tokenize("Hello, world! (yes)") == ["hello", "world", "yes"]


def test_underscores_split_identifiers():
    # choix de conception : response_model -> deux tokens. À toi de trancher
    # et d'adapter ce test si tu décides de garder l'identifiant entier.
    assert tokenize("response_model") == ["response", "model"]


def test_numbers_are_kept():
    assert tokenize("HTTP 404 error") == ["http", "404", "error"]


def test_empty_string():
    assert tokenize("") == []


def test_stopwords_optional():
    assert tokenize("the path of the file") == ["the", "path", "of", "the", "file"]
    assert tokenize("the path of the file", remove_stopwords=True) == ["path", "file"]


def test_unicode_letters_stay_whole():
    assert tokenize("Café naïve Sebastián") == ["café", "naïve", "sebastián"]


def test_symbols_and_emojis_are_separators():
    assert tokenize("fast ✨ api │ docs 🚀") == ["fast", "api", "docs"]


def test_repeated_tokens_are_all_kept():
    # les répétitions doivent être conservées : ce sont elles qui donnent la fréquence
    assert tokenize("a b a") == ["a", "b", "a"]
