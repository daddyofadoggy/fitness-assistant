# fitness-assistant
This a friendly fitness assistant, that uses LLM 
#os.environ['OPENAI_API_KEY'] = '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'

## Project overview

The Fitness Assistant is a RAG application designed to assist users with their fitness routines.

The main use cases include:

1. Exercise Selection: Recommending exercises based on the type
of activity, targeted muscle groups, or available equipment.
2. Exercise Replacement: Replacing an exercise with suitable alternatives.
3. Exercise Instructions: Providing guidance on how to perform a specific exercise.
4. Conversational Interaction: Making it easy to get information without sifting through manuals or websites.

## Dataset

The dataset used in this project contains information about
various exercises, including:

- **Exercise Name:** The name of the exercise (e.g., Push-Ups, Squats).
- **Type of Activity:** The general category of the exercise (e.g., Strength, Mobility, Cardio).
- **Type of Equipment:** The equipment needed for the exercise (e.g., Bodyweight, Dumbbells, Kettlebell).
- **Body Part:** The part of the body primarily targeted by the exercise (e.g., Upper Body, Core, Lower Body).
- **Type:** The movement type (e.g., Push, Pull, Hold, Stretch).
- **Muscle Groups Activated:** The specific muscles engaged during
the exercise (e.g., Pectorals, Triceps, Quadriceps).
- **Instructions:** Step-by-step guidance on how to perform the
exercise correctly.

The dataset was generated using ChatGPT and contains 207 records. It serves as the foundation for the Fitness Assistant's exercise recommendations and instructional support.

You can find the data in [`data/data.csv`](data/data.csv).

## Technologies

- Python 3.9
- Docker and Docker Compose for containerization
- [Minsearch](https://github.com/alexeygrigorev/minsearch) for full-text search
- Flask as the API interface 
- Grafana for monitoring and PostgreSQL as the backend for it
- OpenAI as an LLM

## Running the Application
We use `pipenv` for managing dependencies and Python 3.9
install pipenv 
```bash
pip install pipenv
```
for running notebooks
``` bash 
cd notebooks
pipenv run jupyter notebook
```

## Using the application


### Using `requests`

When the application is running, you can use
[requests](https://requests.readthedocs.io/en/latest/)
to send questions—use [test.py](test.py) for testing it:

```bash
pipenv run python test.py
```

It will pick a random question from the ground truth dataset
and send it to the app.

### CURL

You can also use `curl` for interacting with the API:

```bash
URL=http://127.0.0.1:5000
QUESTION="Is the Lat Pulldown considered a strength training activity, and if so, why?"
DATA='{
    "question": "'${QUESTION}'"
}'

curl -X POST \
    -H "Content-Type: application/json" \
    -d "${DATA}" \
    ${URL}/question
```

You will see something like the following in the response:

```json
{
    "answer": "Yes, the Lat Pulldown is considered a strength training activity. This classification is due to it targeting specific muscle groups, specifically the Latissimus Dorsi and Biceps, which are essential for building upper body strength. The exercise utilizes a machine, allowing for controlled resistance during the pulling action, which is a hallmark of strength training.",
    "conversation_id": "4e1cef04-bfd9-4a2c-9cdd-2771d8f70e4d",
    "question": "Is the Lat Pulldown considered a strength training activity, and if so, why?"
}
```

## Ingestion
## Evaluation

### Retrieval
The basic approach -using minsearch gave the follwing metrices:

* hit_rate: 93.91%
* MRR: 82.64%

The improved version (with better boosting):

* hit_rate: 94.39%
* MRR: 91.07%

The boosting parameters
```python
boost = {
    'exercise_name': 2.452,
    'type_of_activity': 0.094,
    'type_of_equipment': 0.842,
    'body_part': 2.036,
    'type': 1.156,
    'muscle_groups_activated': 0.318,
    'instructions': 0.559
        }
```
### RAG flow
We used LLM-as-a-Judge metric to evaluate the quality of our RAG flow.
We randomly sampled 100 records and found that
* 88 (88%) RELEVENT
* 12 (12%) PARTLY_RELEVENT
* none  NO_RELEVENT

We used `gpt-4o-mini` as LLM

### Monitoring


