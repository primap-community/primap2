"""Tests for _metadata.py"""

import datetime


def test_metadata_properties(opulent_ds):
    ds = opulent_ds
    assert ds.pr.references == "doi:10.1012"
    assert ds.pr.rights == "Use however you want."
    assert ds.pr.contact == "lol_no_one_will_answer@example.com"
    assert ds.pr.title == "Completely invented GHG inventory data"
    assert ds.pr.comment == "GHG inventory data ..."
    assert ds.pr.institution == "PIK"
    assert ds.pr.entity_terminology == "primap2"
    assert ds.pr.publication_date == datetime.date(2099, 12, 31)

    ds.pr.references = "references"
    assert ds.pr.references == "references"
    ds.pr.rights = "rights"
    assert ds.pr.rights == "rights"
    ds.pr.contact = "contact"
    assert ds.pr.contact == "contact"
    ds.pr.title = "title"
    assert ds.pr.title == "title"
    ds.pr.comment = "comment"
    assert ds.pr.comment == "comment"
    ds.pr.institution = "institution"
    assert ds.pr.institution == "institution"
    ds.pr.entity_terminology = "entity_terminology"
    assert ds.pr.entity_terminology == "entity_terminology"
    today = datetime.date.today()
    ds.pr.publication_date = today
    assert ds.pr.publication_date == today
