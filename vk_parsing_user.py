import requests
def take_users():
    token = '4d428b6e4d428b6e4d428b6e274e666c6c44d424d428b6e2a0091d81da35c2108a6468e'
    version = '5.199'
    user_ids = 'princessss_kr'
    posts = []
    response = requests.get('https://api.vk.com/method/users.get',
                            params={
                                'access_token': token,
                                'v': version,
                                'user_ids': user_ids,
                                'fields': 'city, '
                            })
    data = response.json()
    posts.extend(data)  # Добавляем объекты (словарей) в список
    return posts

print(take_users())