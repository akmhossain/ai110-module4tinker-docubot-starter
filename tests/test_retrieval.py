from docubot import DocuBot


def test_auth_token_query_returns_auth_doc():
    bot = DocuBot(docs_folder="docs")
    answer = bot.answer_retrieval_only("Where is the auth token generated?")

    assert "AUTH.md" in answer
    assert "generate_access_token" in answer
