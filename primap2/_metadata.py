import datetime

from . import _accessor_base


class DatasetMetadataAccessor(_accessor_base.BaseDatasetAccessor):
    @property
    def entity_terminology(self) -> str:
        """the terminology of the entity attributes in this dataset's data variables"""
        return self._ds.attrs["entity_terminology"]

    @entity_terminology.setter
    def entity_terminology(self, value: str):
        self._ds.attrs["entity_terminology"] = value

    @property
    def references(self) -> str:
        """citable reference(s) describing the data

        If the references start with ``doi:``, it is a doi, otherwise it is a
        free-form literature reference.
        """
        return self._ds.attrs["references"]

    @references.setter
    def references(self, value: str):
        self._ds.attrs["references"] = value

    @property
    def rights(self) -> str:
        """license or other usage restrictions of the data"""
        return self._ds.attrs["rights"]

    @rights.setter
    def rights(self, value: str):
        self._ds.attrs["rights"] = value

    @property
    def contact(self) -> str:
        """who can answer questions about the data"""
        return self._ds.attrs["contact"]

    @contact.setter
    def contact(self, value: str):
        self._ds.attrs["contact"] = value

    @property
    def title(self) -> str:
        """a succinct description"""
        return self._ds.attrs["title"]

    @title.setter
    def title(self, value: str):
        self._ds.attrs["title"] = value

    @property
    def comment(self) -> str:
        """longer form description"""
        return self._ds.attrs["comment"]

    @comment.setter
    def comment(self, value: str):
        self._ds.attrs["comment"] = value

    @property
    def institution(self) -> str:
        """where the data originates"""
        return self._ds.attrs["institution"]

    @institution.setter
    def institution(self, value: str):
        self._ds.attrs["institution"] = value

    @property
    def publication_date(self) -> datetime.date:
        """The original date of publication of the dataset, if published."""
        return self._ds.attrs["publication_date"]

    @publication_date.setter
    def publication_date(self, value: datetime.date):
        self._ds.attrs["publication_date"] = value
