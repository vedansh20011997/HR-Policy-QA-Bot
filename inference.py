import requests, json

headers = {
    'Content-Type': 'application/json',
}

json_data = {
    'user_query': 'Explain about the esops termination in case of retirement?',
}

response = requests.post('http://localhost:8000/api/ask', headers=headers, json=json_data)

if response.status_code == 200:
    response = response.json()
    print('RESPONSE: ', response['response'])
    print('SOURCES: ', response['sources'])
else:
    print(f'Issue encountered with status code: {response.status_code}')
    print(f'Content: {response.content}')