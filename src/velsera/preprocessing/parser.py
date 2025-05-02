# Parses the cancer text file to extract, id, title and abstract
import logging
import re
from functools import lru_cache

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ResearchPaper(BaseModel):
    id: int | None = Field(..., description="ID of the paper")
    title: str | None = Field(..., description="Title of the paper")
    abstract: str | None = Field(..., description="Abstract of the paper")


class Parser:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    @property
    @lru_cache
    def content(self) -> str:
        with open(self.file_path, "r") as f:
            return f.read()

    def extract_pattern(self, pattern: str) -> str | None:
        match = re.search(pattern, self.content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        else:
            return None

    @property
    def id(self) -> str | None:
        pattern = r"^<ID:(\d+)>$"
        return self.extract_pattern(pattern)

    @property
    def title(self) -> str | None:
        pattern = r"^Title: (.+)$"
        return self.extract_pattern(pattern)

    @property
    def abstract(self) -> str | None:
        pattern = r"^Abstract: (.+)$"
        return self.extract_pattern(pattern)

    def parse(self) -> ResearchPaper | None:
        return ResearchPaper(
            id=int(self.id) if self.id else None,
            title=self.title,
            abstract=self.abstract,
        )


if __name__ == "__main__":

    parser = Parser("Dataset/Cancer/36931075.txt")
    print(parser.parse())
