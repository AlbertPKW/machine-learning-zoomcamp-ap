import requests

#url = 'http://localhost:9797/predict'
url = 'https://wandering-meadow-4571.fly.dev/predict'

employee = {
    'department': 'Technology',
    'region': 2,
    'education': "Bachelor's",
    'gender': 'Male',
    'recruitment_channel': 'Sourcing',
    'no_of_trainings': 1,
    'age': 38,
    'previous_year_rating': 3.0,
    'length_of_service': 9,
    'kpis_met_>80%': 1,
    'awards_won?': 0,
    'avg_training_score': 82
}

response = requests.post(url, json=employee)
predictions = response.json()

print(predictions)
if predictions['is_promoted']:
    print('Employee is likely to be promoted, send email to HR')
else:
    print('Employee is not likely to be promoted')