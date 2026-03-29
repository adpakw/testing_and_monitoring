from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    age: int | None = Field(default=None, description="Возраст")
    workclass: str | None = Field(default=None, description="Тип занятости")
    fnlwgt: int | None = Field(default=None, description="Вес наблюдения")
    education: str | None = Field(default=None, description="Образование")
    education_num: int | None = Field(
        default=None, alias="education.num", description="Уровень образования"
    )
    marital_status: str | None = Field(
        default=None, alias="marital.status", description="Семейное положение"
    )
    occupation: str | None = Field(default=None, description="Профессия")
    relationship: str | None = Field(default=None, description="Роль в семье")
    race: str | None = Field(default=None, description="Раса")
    sex: str | None = Field(default=None, description="Пол")
    capital_gain: int | None = Field(
        default=None, alias="capital.gain", description="Доход от капитала"
    )
    capital_loss: int | None = Field(
        default=None, alias="capital.loss", description="Убытки от капитала"
    )
    hours_per_week: int | None = Field(
        default=None, alias="hours.per.week", description="Часов в неделю"
    )
    native_country: str | None = Field(
        default=None, alias="native.country", description="Страна происхождения"
    )

    class Config:
        allow_population_by_field_name = True


class PredictResponse(BaseModel):
    prediction: int
    probability: float


class UpdateModelRequest(BaseModel):
    run_id: str = Field(min_length=1)


class UpdateModelResponse(BaseModel):
    run_id: str
