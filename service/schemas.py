from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EvidenceFrame(StrictModel):
    timestamp_sec: float = Field(ge=0)
    image_base64: str = Field(min_length=1, max_length=20_000_000)


class Violation(StrictModel):
    type: str = Field(min_length=1, max_length=1000)
    timecode_sec: float = Field(ge=0)
    description: str = Field(min_length=1, max_length=20000)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)


class Analysis(StrictModel):
    verdict: Literal["violation", "no_violation"]
    violations: list[Violation] = Field(max_length=100)

    @model_validator(mode="after")
    def consistent_verdict(self):
        if (self.verdict == "violation") != bool(self.violations):
            raise ValueError("verdict does not match violations")
        return self


class Accepted(StrictModel):
    job_id: str


class Processing(Accepted):
    status: Literal["processing"] = "processing"


class Done(Analysis, Accepted):
    status: Literal["done"] = "done"


class Failed(Accepted):
    status: Literal["failed"] = "failed"
    error: str


JobResponse = Annotated[Processing | Done | Failed, Field(discriminator="status")]
