import pickle
import string

# Original input data.
with open("./test/fixtures/simplices.pkl", "rb") as f:
    # List of strings
    simplices = pickle.load(f)


def letter_simplices_to_numbers(all_simplices):
    total = []
    for simplices in all_simplices:
        simplex_res = []
        for simplex in simplices:
            s_res = []
            for s in simplex:
                s_res.append(string.ascii_lowercase.index(s))
            simplex_res.append(s_res)
        total.append(simplex_res)
    return total


def number_simplices_to_letters(all_simplices):
    total = []
    for simplices in all_simplices:
        simplex_res = []
        for simplex in simplices:
            s_res = []
            for s in simplex:
                s_res.append(string.ascii_lowercase[s])
            simplex_res.append("".join(s_res))
        total.append(simplex_res)
    return total


if __name__ == "__main__":
    numbers = letter_simplices_to_numbers(simplices)
    simplices_letters = number_simplices_to_letters(numbers)
    print(simplices_letters)
