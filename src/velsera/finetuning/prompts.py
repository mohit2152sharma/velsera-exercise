import textwrap

from pydantic import BaseModel, computed_field


class BasePrompt(BaseModel):
    title: str
    abstract: str

    @computed_field
    @property
    def base_prompt(self) -> str:
        string = """
        Given the title and abstract of a research paper 

        Title: {title}
        Abstract: {abstract}

        Is the paper related with cancer? Answer with yes or no only.
        Answer: """
        return textwrap.dedent(string.format(title=self.title, abstract=self.abstract))


class TrainingPrompt(BasePrompt):
    label: str

    @computed_field
    @property
    def prompt(self) -> str:
        return (self.base_prompt + f"""{self.label}""").strip()


class TestingPrompt(BasePrompt):
    @computed_field
    @property
    def prompt(self) -> str:
        return self.base_prompt.strip()


if __name__ == "__main__":
    prompt = TestingPrompt(title="Title", abstract="Abstract")
    print(prompt.prompt)
