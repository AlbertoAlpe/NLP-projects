import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor
import random
from collections import defaultdict

# Assicurarsi di avere i corpora necessari
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('semcor')

def lesk_algorithm(context_sentence, ambiguous_word):
    max_overlap = 0
    best_sense = None
    context = set(nltk.word_tokenize(context_sentence))
    
    for sense in wn.synsets(ambiguous_word, pos=wn.NOUN):
        signature = set(nltk.word_tokenize(sense.definition()))
        for example in sense.examples():
            signature.update(nltk.word_tokenize(example))
        
        overlap = len(context.intersection(signature))
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    
    return best_sense

def evaluate_lesk(sentences, target_words, annotated_senses):
    correct = 0
    for i in range(len(sentences)):
        context_sentence = ' '.join(word for word in sentences[i])
        ambiguous_word = target_words[i]
        predicted_sense = lesk_algorithm(context_sentence, ambiguous_word)
        actual_sense = annotated_senses[i]
        
        if predicted_sense == actual_sense:
            correct += 1
    
    return correct / len(sentences)

# Estrarre 50 frasi casuali da SemCor
def extract_sentences_from_semcor(n):
    sentences = []
    target_words = []
    annotated_senses = []

    for i in range(n):
        sentence = semcor.tagged_sents(tag='both')[i]
        words = []
        word = None
        sense = None

        for w in sentence:
            if isinstance(w, nltk.Tree) and w.label().startswith('NN'):
                word = w.leaves()[0]
                sense = w.label()
                if sense is not None:
                    sense = wn.synset(sense)
                break
            else:
                words.append(w[0])
        
        if word and sense:
            sentences.append(words)
            target_words.append(word)
            annotated_senses.append(sense)
    
    return sentences, target_words, annotated_senses

# Randomizzare la selezione delle frasi e delle parole, e restituire l'accuratezza media
def randomize_evaluation(n, iterations):
    accuracies = []
    for _ in range(iterations):
        sentences, target_words, annotated_senses = extract_sentences_from_semcor(n)
        accuracy = evaluate_lesk(sentences, target_words, annotated_senses)
        accuracies.append(accuracy)
    
    return sum(accuracies) / len(accuracies)

# Eseguire l'algoritmo
num_sentences = 50
iterations = 10
average_accuracy = randomize_evaluation(num_sentences, iterations)
print(f"Average Accuracy over {iterations} iterations: {average_accuracy * 100:.2f}%")
