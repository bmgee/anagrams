#!/usr/bin/env python3

"""
Program for generating all anagrams of a given word and word_list.

Definitions:

    k-divider-word-partition:
        An ordered partitioning of a word into k non empty parts
        (list of strings).

        ex. using word 'ABCC'
            0-divider-word-partitions:
                ['ABCC']
            1-divider-word-partitions:
                ['A', 'BCC'], ['AB', 'CC'], ['ABC', 'C']
            2-divider-word-partitions:
                ['A', 'B', 'CC'], ['A', 'BC', 'C'], ['AB', 'C', C']
            3-divider-word-partitions:
                ['A', 'B', 'C', 'C']

            For a given word and k the number of k-divider-word-partitions is:
               C(len(word), k - 1) i.e. (len(word) choose (k - 1)) if k >= 1
               1 if k < 1

    k-divider:
        A tuple of integers defining a k-divider-word-partition.

        ex. using word 'ABCC' and corresponding to the
        k-divider-word-partitions above.
            0-dividers: ()
            1-dividers: (1,), (2,), (3,)
            2-dividers: (1, 2), (1, 3), (2, 3)
            3-dividers: (1, 2, 3)

    word-partition:
        A non specific k-divider-word-partition.

    divider:
        A non specific k-divider.

    integer-partition:
        For a given number N, a way of writing N as as sum of positive integers
        (tuple).

    anagram:
        With reference to a specific word and word_list, an anagram is a
        word-partition of some permutation of the word with the word_partition
        being a subset of the word_list.
"""

import argparse
import multiprocessing as mp
from functools import partial
from itertools import combinations, permutations, product
from collections import defaultdict

def div_to_integer_partition(divider, word_length):
    """ integer-partition corresponding to given divider and word_length.

    For a given divider and word_length there is a corresponding
    integer-partition.

    The corresponding integer partition is just the differences between the
    consecutive pairwise elements of the augmented_divider where the
    augmented_divider is defined to be the divider with 0 prepended and the
    word_length appended.
    examples:
        divider: (1, 3, 4), word_length: 6
        -> augmented_divider: (0, 1, 3, 4, 6)
        -> integer_partition: (1, 2, 1, 2)

        divider: (2, 5), word_length: 9
        -> augmented_divider: (0, 2, 5, 9)
        -> integer_partition: (2, 3, 4)

    Args:
        divider: k-divider

    Returns:
        integer_partition: integer-partition (tuple)
    """

    augmented_divider = (0,) + divider + (word_length,)

    integer_partition = tuple(augmented_divider[i] - augmented_divider[i - 1]
                              for i in range(1, len(augmented_divider)))

    return integer_partition

def ip_to_word_partitions(integer_partition, word_list_by_length):
    """ word-partitions corresponding to the given integer_partition and
        word_list_by_length.

    Args:
        integer_partition: integer partition
        word_list_by_length: dictionary of lists of strings keyed by length

    Returns:
        wps: iterator of word partitions
    """

    word_choice_bins = tuple(word_list_by_length[j] for j in integer_partition)
    wps = product(*word_choice_bins)

    return wps

def wp_to_comparable_form(word_partition):
    """ A comparable form of the given word_partition.

    The returned comparable form can be compared to the comparable form of the
    word to determine whether the letter usage is the same.

    Args:
        word_partition: word_partition

    Returns:
        comparable_form: sorted string
    """

    comparable_form = ''.join(sorted(''.join(word_partition)))

    return comparable_form

def dividers_to_word_partitions(word, dividers):
    """ word-partitions of a word defined via set of dividers.

    Args:
        word: string
        dividers: tuple of k-dividers

    Returns:
        dividers_word_partitions: list of word-partitions
    """

    dividers_word_partitions = []
    for div in dividers:
        div_word_partitions = []
        if len(div) <= 0:
            # 0-divider correspondends to 0-divider-word-partition
            # which corresponds to 1 item list containing word itself.
            div_word_partitions.append(word)
        else:
            div_word_partitions.append(word[:div[0]])
            div_word_partitions.extend([word[div[i]:div[i + 1]]
                                        for i in range(len(div) - 1)])
            div_word_partitions.append(word[div[-1]:])

        dividers_word_partitions.append(div_word_partitions)

    return dividers_word_partitions

def ordered_word_anagrams(word, word_list_set, dividers):
    """ All word-partitions of a word defined via set of dividers,
        filtered down to word-partitions where words found in word_list_set.

    Args:
        word: string
        word_list_set: set of strings
        dividers: tuple of k-dividers

    Returns:
        owa: list of word-partitions
    """

    owa = [wp for wp in dividers_to_word_partitions(word, dividers)
           if word_list_set.issuperset(wp)]

    return owa

def div_to_anagrams(divider, word_list_by_length, word_length,
                    comparable_form_of_word):
    """ word-partitions defined by a divider and word_list_by_length,
        filtered down to those which are anagrams of a word
        (given in comparable form).

    Note: Instead of take the word as an an argument, since we don't need it
        directly, we take word_length and comparable_form_of_word instead.
        This saves computing these with every call to this function.

    Args:
        divider: k-divider
        word_list_by_length: dictionary of lists of strings keyed by length
        word_length: precomputed length of word
        comparable_form_of_word: precomputed comparable form of word

    Returns:
        dta: iterable of word-partitions
    """

    # Each divider corresponds to an integer partition of the word length.
    word_length_int_partition = div_to_integer_partition(divider, word_length)

    # A length partition defines a way to divy up the word into component sizes.
    # Figure out possible word partitions based on these.
    ip_word_partitions = ip_to_word_partitions(word_length_int_partition,
                                               word_list_by_length)

    dta = [wp for wp in ip_word_partitions
           if wp_to_comparable_form(wp) == comparable_form_of_word]

    return dta

def reduced_word_list(word, word_list):
    """ Set of words in word_list that are a subsequence of any permutation of
        the given word.

    Args:
        word: string
        word_list: list of strings

    Returns:
        rwls: frozenset of strings
    """

    rwls = []
    for i in range(1, len(word)):
        i_permutations = frozenset(''.join(w) for w in permutations(word, i))
        rwls.extend(i_permutations.intersection(word_list))

    return frozenset(rwls)

def anagrams_word_centric(word, word_list_set, dividers,
                          nprocs=mp.cpu_count, chunk_size=8):
    """ All word-partitions of all permutations of a word,
        filtered down to word-partitions where words found in word_list
        (i.e. anagrams of given word and word_list_set).
        Computation focused on permutations of word given.

        Anagram candidates are generated from the word and its permutations and
        then compared against the words in the word_list_set to see if they are
        anagrams (is anagram if the strings of the anagram candidate are
        contained in word_list_set).

    Args:
        word: string
        word_list_set: set of strings
        nprocs: max number of processes to use for computation
        chunk_size: guessed chunk size for multiprocessing for good performance

    Returns:
        pop: list of partitions, each partition is a iterable of strings
    """

    word_permutations = (''.join(wp) for wp in permutations(word))

    # Create single argument version of the ordered_word_anagrams function
    # with preset word_list_set and dividers arguments.
    owas = partial(ordered_word_anagrams, word_list_set=word_list_set,
                   dividers=dividers)

    if nprocs <= 1:
        owag = (owas(wp) for wp in word_permutations)
        pop = [item for sublist in owag for item in sublist]
    else:
        with mp.Pool(nprocs) as pool:
            owag = pool.imap_unordered(owas, word_permutations, chunk_size)
            # Flatten owag.
            pop = [item for sublist in owag for item in sublist]

    return pop

def anagrams_word_list_centric(word, word_list_set, dividers,
                               nprocs=mp.cpu_count, chunk_size=1):
    """ All word-partitions of all permutations of a word,
        filtered down to word-partitions where words found in word_list
        (i.e. anagrams of given word and word_list_set).
        Computation focused on permutations of suitable words in the
        given word_list_set.

        Anagram candidates are generated from the word_list_set and then
        compared against the word to see if they are anagrams
        (is anagram if the anagram candidate has the same letter usage
        as the word).

    Args:
        word: string
        word_list_set: set of strings
        nprocs: max number of processes to use for computation
        chunk_size: guessed chunk size for multiprocessing for good performance

    Returns:
        pop: list of partitions, each partition is a iterable of strings
    """

    # comparable form of the word given.
    # single element list containing word string is the word partition for
    # the word, which is the argument type needed for the function call.
    comparable_form_of_word = wp_to_comparable_form([word])

    # Bucket the words in word_list_set by their length.
    word_list_by_length = defaultdict(list)
    for wl_word in word_list_set:
        word_list_by_length[len(wl_word)].append(wl_word)

    # Create single argument version of the div_to_anagrams function
    # with preset word_list_by_length, word_length, comparable_form_of_word.
    dtas = partial(div_to_anagrams, word_list_by_length=word_list_by_length,
                   word_length=len(word),
                   comparable_form_of_word=comparable_form_of_word)

    if nprocs <= 1:
        dtag = (dtas(d) for d in dividers)
        pop = [item for sublist in dtag for item in sublist] # Flatten dtag
    else:
        with mp.Pool(nprocs) as pool:
            dtag = pool.imap_unordered(dtas, dividers, chunk_size)
            # Flatten dtag.
            pop = [item for sublist in dtag for item in sublist]

    return pop

def word_anagrams(word, word_list,
                  method='word_list_centric',
                  nprocs=mp.cpu_count()):
    """ All word-partitions of all permutations of a word,
        filtered down to word-partitions where words found in word_list
        (i.e. anagrams of given word and word_list_set).

    Note: word and word_list case sensitive.

    Args:
        word: string
        word_list: list of strings
        method: string indicating by what method we compute anagrams
            word_centric: generate anagrams focusing computation on
                permutations of the word given
            word_list_centric: generate anagrams focusing computation on
                permutations of suitable words in the given word_list
        nprocs: max number of processes to use for computation

    Returns:
        pop: list of partitions, each partition is a iterable of strings
    """

    # Remove all white space from word.
    word = ''.join(word.split())

    # Reduce word_list_set to words that are a subsequence of any
    # permutation of word.
    word_list_set = reduced_word_list(word, word_list)

    # Tuple of size len(word) containing the 0-divider-word-partitions,
    # 1-divider-word-partitions,..., (len(word) - 1)-divider-word-partitions
    # of the given word.
    k_divider_sets = tuple(c for c in tuple(combinations(range(1, len(word)), k)
                                            for k in range(len(word))))

    # Flattened version of the k_divider_sets iterable.
    dividers = tuple(item for sublist in k_divider_sets for item in sublist)

    anagram_methods = {'word_centric': anagrams_word_centric,
                       'word_list_centric': anagrams_word_list_centric}

    anagram_method = anagram_methods[method]

    pop = anagram_method(word, word_list_set, dividers, nprocs)

    return pop

def main():
    """ main.
    """

    parser = argparse.ArgumentParser(description='Generate anagrams.')
    parser.add_argument('word',
                        help='Word to find anagrams of.')
    parser.add_argument('word_list_file',
                        help=('UTF-8 text file defining the universe of words '
                              '(a word dictionary) delimited by newlines.'))
    parser.add_argument('out_file',
                        help=('Output file results are saved to '
                              '(UTF-8 text file delimited by newlines).'))
    parser.add_argument('--method', '-m',
                        help=('Method of computation. '
                              'word_centric: Anagram candidates generated from '
                              'word and checked against word_list_file. '
                              'word_list_centric: Anagram candidates generated '
                              'from words_file and checked against word.'),
                        choices=['word_centric', 'word_list_centric'],
                        default='word_list_centric')
    parser.add_argument('--nprocs', '-n',
                        help=('Max number of processes for computation.'),
                        default=mp.cpu_count())

    args = parser.parse_args()

    word = args.word

    with open(args.word_list_file, mode='rt', encoding='utf-8') as words_file:
        word_list = words_file.read().splitlines()

    method = args.method
    nprocs = args.nprocs

    anagrams = word_anagrams(word, word_list, method, nprocs)

    # Convert list of word_partitions to list of strings
    anagrams = [' '.join(anagram) for anagram in anagrams]

    anagrams.sort()

    with open(args.out_file, mode='xt', encoding='utf-8') as out_file:
        num_chars_written = out_file.write('\n'.join(anagrams))

    return num_chars_written

if __name__ == '__main__':
    main()
