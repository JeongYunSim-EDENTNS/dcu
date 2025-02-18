FROM python:3.10

WORKDIR /app

RUN pip install poetry==1.5.1

RUN poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* ./

RUN poetry lock --no-update
RUN poetry install --no-interaction --no-ansi --no-root --no-directory

COPY . .

CMD exec python app.py
