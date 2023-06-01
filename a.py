import numpy

# average_performances = numpy.load(("combined/average_performances_small.npy"), allow_pickle=True)
# average_performances = average_performances.item()
# print("------------------------------------------")
# print(average_performances["training"]["accuracy_target_model"])
# print()
average_performances = numpy.load(("average_performances.npy"), allow_pickle=True)
average_performances = average_performances.item()
print(average_performances)
print("------------------------------------------")