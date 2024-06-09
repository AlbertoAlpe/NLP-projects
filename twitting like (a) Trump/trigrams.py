import os

###LETTURA FILE .CSV
file_name = 'tweets.csv'
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'trump_twitter_archive/', file_name)

with open(file_path, 'r', encoding='utf-8') as train:
   prime_10_righe = train.readlines()[:10]

for riga in prime_10_righe:
   riga = riga.split()
