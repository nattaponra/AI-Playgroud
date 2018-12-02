import math


w11_2 =  0.33804757
w21_2 =  0.42412811

w12_2 = -0.48693727
w22_2 = -0.33546053

w13_2 =  0.38398597
w23_2 = -0.86907973 


w11_3 =  0.29484888
w21_3 = -0.97796236
w31_3 = -0.61818333

def sigmod(input):
     return 1 / (1 + math.exp(-input))


#Forward pass
#Step1 Input layer 
print("######[Forward pass]#####  Step1 Input layer ##############")
x1 = 1
x2 = 1
y  = 0

training_data = [[1,1],[1,0],[0,1],[0,0]]
output_data   = [1,1,1,0]
for iteration in range(1,10000):
    for i in range(len(training_data)):
        print("Iteration:"+str(iteration+1))
        x1 = training_data[i][0]
        x2 = training_data[i][1]
        y =  output_data[i]
        #Step2 Hidden layer
        #print("##############   Step2 Hidden layer ##############")
        a1_2 = sigmod(w11_2*x1+ w21_2*x2)
        a2_2 = sigmod(w12_2*x1+ w22_2*x2)
        a3_2 = sigmod(w13_2*x1+ w23_2*x2)

        #print(a1_2)
        #print(a2_2)
        #print(a3_2)


        #Step 3) Output layer
        #print("############## Step 3) Output layer ##############")
        a1_3 = sigmod(w11_3*a1_2 + w21_3*a2_2 + w31_3*a3_2)

        #print(a1_3)

        #print("############## Step 4) Calculate the cost ##############")
        #print("Expected Output is:",y)
        #print("Actual Output is:",a1_3)
        E =  ((y-a1_3)*(y-a1_3))/2
        #print("Error:",E)


        #print("###[Backpropagation pass]### Step 5) Error in the output layer ##############")
        error1_3 = (y-a1_3)*a1_3*(1-a1_3)
        #print("Error in the output layer is:",error1_3)


        #print("############ 6) Error in the hidden layer##############")
        error1_2 = (error1_3*w11_3)*a1_2 *(1-a1_2)
        error2_2 = (error1_3*w21_3)*a2_2 *(1-a2_2)
        error3_2 = (error1_3*w31_3)*a3_2 *(1-a3_2)
        #print("Error in the hiden layer a1 is:",error1_2)
        #print("Error in the hiden layer a2 is:",error2_2)
        #print("Error in the hiden layer a3 is:",error3_2)

        #print("############ Step 7) Calculate the error with respect to weights between hidden and output layer ##############")
        error_respect_w11_3 = a1_2*error1_3
        error_respect_w21_3 = a2_2*error1_3
        error_respect_w31_3 = a3_2*error1_3
        #print(error_respect_w11_3)
        #print(error_respect_w21_3)
        #print(error_respect_w31_3)

        #print("############ Step 8) Calculate the error with respect to weights between input and hidden layer ##############")
        error_respect_w11_2 = x1*error1_2
        error_respect_w12_2 = x1*error2_2

        error_respect_w13_2 = x1*error3_2
        error_respect_w21_2 = x2*error1_2

        error_respect_w22_2 = x2*error2_2
        error_respect_w23_2 = x2*error3_2

        #print(error_respect_w11_2)
        #print(error_respect_w12_2)
        #print(error_respect_w13_2)
        #print(error_respect_w21_2)
        #print(error_respect_w22_2)
        #print(error_respect_w23_2)


        #print("############ Step 9) Update the weights between hidden and output layer ##############")

        w11_3_new = w11_3 + error_respect_w11_3
        w21_3_new = w21_3 + error_respect_w21_3
        w31_3_new = w31_3 + error_respect_w31_3
        w11_3 = w11_3_new
        w21_3 = w21_3_new
        w31_3 = w31_3_new
        # print("new weight w11_3:",w11_3_new)
        # print("new weight w21_3:",w21_3_new)
        # print("new weight w31_3:",w31_3_new)


        #print("############ Step 10) Update the weights between input and hidden layer ##############")


        w11_2_new = w11_2 + error_respect_w11_2
        w12_2_new = w12_2 + error_respect_w12_2
        w13_2_new = w13_2 + error_respect_w13_2
        w21_2_new = w21_2 + error_respect_w21_2
        w22_2_new = w22_2 + error_respect_w22_2
        w23_2_new = w23_2 + error_respect_w23_2


        w11_2  = w11_2_new
        w12_2  = w12_2_new
        w13_2  = w13_2_new
        w21_2  = w21_2_new
        w22_2  = w22_2_new
        w23_2  = w23_2_new


        # print("new weight w11_2:",w11_2_new)
        # print("new weight w12_2:",w12_2_new)
        # print("new weight w13_2:",w13_2_new)
        # print("new weight w21_2:",w21_2_new)
        # print("new weight w22_2:",w22_2_new)
        # print("new weight w23_2:",w23_2_new)

        # print("####################################################")
        # print("Output:",a1_3)
        # print("####################################################")


#Test Model

test_data = [[1,1],[1,0],[0,1],[0,0]]
for data in test_data:
 
    x1 =  data[0] 
    x2 =  data[1]

    a1_2 = sigmod(w11_2*x1+ w21_2*x2)
    a2_2 = sigmod(w12_2*x1+ w22_2*x2)
    a3_2 = sigmod(w13_2*x1+ w23_2*x2)
    a1_3 = sigmod(w11_3*a1_2 + w21_3*a2_2 + w31_3*a3_2)
    print(str(data[0])+" XOR "+str(data[1])+"="+str(a1_3))