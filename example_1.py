#!/usr/bin/env python3

"""
    Example usage of anagrams.py
"""

from multiprocessing import cpu_count
from anagrams import word_anagrams

def anagram_prob_01(word, word_list, method, nprocs):
    """ anagram problem 01.

    The problem is - given a word from a wordlist, find one anagram with the
    most number of words (from the given wordlist) and one anagram with just
    two words in them (if one exists). If no anagrams exist with two or more
    words, the program should print an empty string for both cases.

    For example:
    For the word incredible, "bile cinder" is one anagram with just two words.
    For the word infinite, "net if I in" is one anagram with the most number
    of words.
    (you can just print one of the many, if there are many such with the same
     number of words).

    Args:
        word: string from word_list
        word_list: list of possible words
        method: string indicating by what method we compute anagrams
            word_centric: generate anagrams focusing computation on
                permutations of the word given
            word_list_centric: generate anagrams focusing computation on
                permutations of suitable words in the given word_list
        nprocs: max number of processes to use for computation

    Returns:
        anagram string containing the most words, anagram string containing two
        words
    """

    anagrams = word_anagrams(word, word_list, method, nprocs)

    single_two_word_anagrams = (pa for pa in anagrams if len(pa) == 2)

    single_two_word_anagram = next(single_two_word_anagrams, False)

    if single_two_word_anagram:
        single_most_words_anagram = max(anagrams, key=len)
        return (' '.join(single_most_words_anagram),
                ' '.join(single_two_word_anagram))
    else:
        return ('', '')

def main():
    """ main.
    """

    word = 'infinite'

    # Word list from
    # https://raw.githubusercontent.com/dwyl/english-words/master/words.txt
    with open('words.txt', 'r') as words_file:
        word_list = words_file.read().splitlines()

    method = 'word_list_centric'
    nprocs = cpu_count()

    return anagram_prob_01(word, word_list, method, nprocs)

if __name__ == '__main__':
    print(main())
