#! /user/bin/env python2

# Import required packages
import pandas as pd

# Weights and bias
WEIGHT_1 = 1
WEIGHT_2 = 1
BIAS = -2

# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]

# For storing the results
outputs = []

for test_input, correct_output in zip(test_inputs, correct_outputs):
    # Calculating the linear combination value
    linear_combination = WEIGHT_1 * test_input[0] + WEIGHT_2 * test_input[1] + BIAS
    # Heaviside step function
    output = int(linear_combination >= 0)
    output_is_correct = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, output_is_correct])

print ("Calculated Outputs")
print (outputs)

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))