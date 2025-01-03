import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from stop_words import get_stop_words
import numpy as np
import time

# Константы
TOKEN = '4d428b6e4d428b6e4d428b6e274e666c6c44d424d428b6e2a0091d81da35c2108a6468e'
VERSION = '5.199'

# Функция для безопасного выполнения запросов к API
def safe_request(url, params, retries=3, delay=0.5):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if 'error' in data and data['error']['error_code'] == 6:
                print("Слишком много запросов. Повторная попытка...")
                time.sleep(delay * (attempt + 1))
                continue
            return data
        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса: {e}")
            time.sleep(delay * (attempt + 1))
    return {"error": {"error_msg": "Превышено количество попыток.", "error_code": 6}}

# Функция для получения ID пользователя по его имени
def get_user_id(user_name):
    params = {
        'access_token': TOKEN,
        'v': VERSION,
        'user_ids': user_name
    }
    data = safe_request('https://api.vk.com/method/users.get', params)
    if 'response' in data:
        user_id = data['response'][0]['id']
        return user_id  # Возвращаем числовой ID
    else:
        print(f"Ошибка API: {data['error']['error_msg']} (Код: {data['error']['error_code']})")
    return None

# Функция для получения постов пользователя
def take_user_posts(user_id):
    posts = []
    offset = 0
    while True:
        params = {
            'access_token': TOKEN,
            'v': VERSION,
            'owner_id': user_id,
            'count': 100,
            'offset': offset
        }
        data = safe_request('https://api.vk.com/method/wall.get', params)
        if 'error' in data:
            print(f"Ошибка API: {data['error']['error_msg']} (Код: {data['error']['error_code']})")
            break
        items = data.get('response', {}).get('items', [])
        if not items:
            break
        posts.extend([post.get('text', '').strip() for post in items if 'text' in post])
        offset += 100
        time.sleep(0.5)
    return posts

# Функция для получения подписок пользователя
def get_user_subscriptions(user_id):
    groups = []
    offset = 0
    while True:
        params = {
            'access_token': TOKEN,
            'v': VERSION,
            'user_id': user_id,
            'count': 200,
            'offset': offset
        }
        data = safe_request('https://api.vk.com/method/users.getSubscriptions', params)
        if 'error' in data:
            print(f"Ошибка API: {data['error']['error_msg']} (Код: {data['error']['error_code']})")
            break
        items = data.get('response', {}).get('items', [])
        if not items:
            break
        groups.extend([item['name'] for item in items if item.get('type') in ['page', 'group']])
        offset += 200
        time.sleep(0.5)
    return groups

# Функция для анализа текстов постов
def analyze_texts(texts, vectorizer=None):
    russian_stop_words = get_stop_words('russian')
    texts = [text for text in texts if text.strip()]
    if not texts:
        raise ValueError("Нет текстов для анализа после фильтрации.")
    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words=russian_stop_words, max_features=500)
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = vectorizer.transform(texts)
    return tfidf_matrix, vectorizer

# Функция для построения среднего профиля
def build_average_profile(users_posts, users_subscriptions):
    all_texts = [text for posts in users_posts for text in posts]
    if not all_texts:
        raise ValueError("Нет текстов для построения профиля.")
    tfidf_matrix, vectorizer = analyze_texts(all_texts)

    all_groups = [group for groups in users_subscriptions for group in groups]
    group_counter = Counter(all_groups)

    return {
        'tfidf_matrix': tfidf_matrix.mean(axis=0),
        'vectorizer': vectorizer,
        'group_stats': group_counter
    }

# Функция для сравнения нового пользователя
def compare_user_with_profile(user_posts, user_subscriptions, avg_profile):
    if not user_posts:
        raise ValueError("Нет текстов для анализа нового пользователя.")

    tfidf_matrix, _ = analyze_texts(user_posts, vectorizer=avg_profile['vectorizer'])
    tfidf_matrix_array = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    text_similarity = cosine_similarity([tfidf_matrix_array], [np.asarray(avg_profile['tfidf_matrix']).flatten()])[0][0]

    group_similarity = sum(avg_profile['group_stats'].get(group, 0) for group in user_subscriptions) / max(1, len(avg_profile['group_stats']))

    return {
        'text_similarity': text_similarity,
        'group_similarity': group_similarity
    }

# Основной блок
if __name__ == "__main__":
    user_names = ['princessss_kr', 'id175633281', 'ropik05']
    all_posts = []
    all_subscriptions = []

    for user_name in user_names:
        user_id = get_user_id(user_name)
        if user_id:
            user_posts = take_user_posts(user_id)
            user_subscriptions = get_user_subscriptions(user_id)
            all_posts.append(user_posts)
            all_subscriptions.append(user_subscriptions)

    try:
        avg_profile = build_average_profile(all_posts, all_subscriptions)
        print("Средний профиль успешно построен.")
    except ValueError as e:
        print(f"Ошибка при построении среднего профиля: {e}")
        exit()

    new_user_name = 'ropik05'
    new_user_id = get_user_id(new_user_name)
    if new_user_id:
        new_user_posts = take_user_posts(new_user_id)
        new_user_subscriptions = get_user_subscriptions(new_user_id)

        try:
            comparison = compare_user_with_profile(new_user_posts, new_user_subscriptions, avg_profile)
            print("Сравнение нового пользователя:", comparison)
        except ValueError as e:
            print(f"Ошибка анализа данных для нового пользователя: {e}")
