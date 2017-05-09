#
#	Author	:	Tarun Jain
#	Topic	:	MNIST Model
#	Date	:	03-May-2017 - 04-May-17
#


#
#	Imports 
#
from matplotlib import pyplot
import numpy as np
import os
import shutil
import time
from caffe2.python import core, cnn, net_drawer, workspace, visualize

print("\nNecessities imported!")

#
#	Directory Setup and Logging
#

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=-1'])

# set this where the root of caffe2 is installed
caffe2_root = "/home/tarun/downloads/new_caffe2"


# Set Addresses for various directory or files
current_folder = os.getcwd()

data_folder = os.path.join(current_folder, 'tutorial_data', 'mnist')
root_folder = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')
image_file_train = os.path.join(data_folder, "train-images-idx3-ubyte")
label_file_train = os.path.join(data_folder, "train-labels-idx1-ubyte")
image_file_test = os.path.join(data_folder, "t10k-images-idx3-ubyte")
label_file_test = os.path.join(data_folder, "t10k-labels-idx1-ubyte")
leveldb_train_folder = os.path.join(data_folder, 'mnist-train-nchw-leveldb')
leveldb_test_folder = os.path.join(data_folder, 'mnist-test-nchw-leveldb')

#
#	Downloading Data
#

# Get the dataset if it is missing
def DownloadDataset(url, path):
    import requests, zipfile, StringIO
    print "\nDownloading... ", url, " to ", path
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall(path)

# Create Data Directory if not exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Download Train Data if not available
if not os.path.exists(label_file_train):
    DownloadDataset("https://s3.amazonaws.com/caffe2/datasets/mnist/mnist.zip", data_folder)

# Generate Leveldb data -- to generate lmdb ; change the flag is syscall string
def GenerateDB(image, label, name):
    name = os.path.join(data_folder, name)
    print '\nDB name: ', name
    syscall ="/home/tarun/downloads/new_caffe2/build/caffe2/binaries/make_mnist_db  --channel_first --db leveldb --image_file " + image + " --label_file " + label + " --output_file " + name
    print "\nCreating database with: ", syscall
    os.system(syscall)

# Generate the leveldb database
if not os.path.exists(leveldb_train_folder):
	GenerateDB(image_file_train, label_file_train, "mnist-train-nchw-leveldb")

if not os.path.exists(leveldb_test_folder):
	GenerateDB(image_file_test, label_file_test, "mnist-test-nchw-leveldb")

# Clear the directory tree specified by root_folder if exists
if os.path.exists(root_folder):
    print("\nLooks like you ran this before, so we need to cleanup those old workspace files...")
    shutil.rmtree(root_folder)

# Create Root Folder
os.makedirs(root_folder)

# Reset Workspace
workspace.ResetWorkspace(root_folder)

# Print Directory Addresses
print("\ntraining data folder:"+data_folder)
print("\nworkspace root folder:"+root_folder)


#
#	Add Input Function
#
def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size=batch_size,db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label

print("\nInput function created.")

#
#	Create Model
#
def AddLeNetModel(model, data):
    conv1 = model.Conv(data, 'conv1', 1, 20, 5)
    pool1 = model.MaxPool(conv1, 'pool1', kernel=2, stride=2)
    conv2 = model.Conv(pool1, 'conv2', 20, 50, 5)
    pool2 = model.MaxPool(conv2, 'pool2', kernel=2, stride=2)
    fc3 = model.FC(pool2, 'fc3', 50 * 4 * 4, 500)
    fc3 = model.Relu(fc3, fc3)
    pred = model.FC(fc3, 'pred', 500, 10)
    softmax = model.Softmax(pred, 'softmax')
    return softmax

print("\nModel function created.")

#
#	Add Accuracy Function
#
def AddAccuracy(model, softmax, label):
    accuracy = model.Accuracy([softmax, label], "accuracy")
    return accuracy

print("\nAccuracy function created.")

#
#	Add training Function
#

def AddTrainingOperators(model, softmax, label):
    
    # something very important happens here
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    
    # track the accuracy of the model
    AddAccuracy(model, softmax, label)
    
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    
    #
    # do a simple stochastic gradient descent
    #
    
    # Counter for No. of Iterations in Training
    ITER = model.Iter("iter")
    
    # set the learning rate schedule
    LR = model.LearningRate(ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
    
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        
	# Note how we get the gradient of each parameter - CNNModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        
	# The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)
    
    # let's checkpoint every 20 iterations, which should probably be fine.
    # you may need to delete tutorial_files/tutorial-mnist to re-run the tutorial
    model.Checkpoint([ITER] + model.params, [], db="mnist_lenet_checkpoint_%05d.leveldb", db_type="leveldb", every=1000)

print("\nTraining function created.")

#
#	Add Bookkeeping Operations
#

def AddBookkeepingOperators(model):
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
    # Now, if we really want to be verbose, we can summarize EVERY blob
    # that the model produces; it is probably not a good idea, because that
    # is going to take time - summarization do not come for free. For this
    # demo, we will only show how to summarize the parameters and their
    # gradients.

print("\nBookkeeping function created")

#
#	Train, Test, Deploy Model
#

# Train Model
train_model = cnn.CNNModelHelper(order="NCHW", name="mnist_train")

# Train Batch Size = 64
train_batch_size = 64
data, label = AddInput(train_model, batch_size=train_batch_size, db=os.path.join(data_folder, 'mnist-train-nchw-leveldb'), db_type='leveldb')
softmax = AddLeNetModel(train_model, data)
AddTrainingOperators(train_model, softmax, label)
AddBookkeepingOperators(train_model)

# Testing model. 
# We will set the batch size to 100, so that the testing pass is 100 iterations (10,000 images in total).
# For the testing model, we need the data input part, the main LeNetModel part, and an accuracy part. 
# Note that init_params is set False because we will be using the parameters obtained from the train model.
test_model = cnn.CNNModelHelper(order="NCHW", name="mnist_test", init_params=False)
# Test Batch Size = 100
test_batch_size = 100
data, label = AddInput(test_model, batch_size=test_batch_size, db=os.path.join(data_folder, 'mnist-test-nchw-leveldb'), db_type='leveldb')
softmax = AddLeNetModel(test_model, data)
AddAccuracy(test_model, softmax, label)

# Deployment model. We simply need the main LeNetModel part.
deploy_model = cnn.CNNModelHelper(order="NCHW", name="mnist_deploy", init_params=False)
AddLeNetModel(deploy_model, "data")
# You may wonder what happens with the param_init_net part of the deploy_model.
# No, we will not use them, since during deployment time we will not randomly
# initialize the parameters, but load the parameters from the db.

print('\nCreated training and deploy models.')


#
#	CNNModelHelper class has not executed anything yet. All it does is to declare the network, which is basically creating the protocol buffers.
#	Dump All Protobufs in root_folder
#
with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fid:fid.write(str(train_model.net.Proto()))
with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fid:fid.write(str(train_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fid:fid.write(str(test_model.net.Proto()))
with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fid:fid.write(str(test_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:fid.write(str(deploy_model.net.Proto()))

print("\nProtocol buffers files have been created in your root folder: "+root_folder)

#
#	Running training model
#

# Initialize training net once
workspace.RunNetOnce(train_model.param_init_net)

# Create Network -  puts the actual network generated from the protobuf into the workspace.
workspace.CreateNet(train_model.net)

# Set No. of Iterations
total_train_iters = 10000

# Numpy Array to record accuracy and loss of each iterations
accuracy = np.zeros(total_train_iters)
loss = np.zeros(total_train_iters)

print("\nBegin Training ...\n")
start_train = time.time()
# Run training for total_iters times, fetch accuracy and loss of each iteration
for i in range(total_train_iters):
    workspace.RunNet(train_model.net.Proto().name)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')
    print "Training Iteration: #" + str(i) + " Accuracy: " + str(accuracy[i]) + " Loss: " + str(loss[i])

end_train= time.time()

#
#	Testing Model
#

# run a test pass on the test net
workspace.RunNetOnce(test_model.param_init_net)

# Create Test Net
workspace.CreateNet(test_model.net)

# Set No. of iteration for test
total_test_iters = 100
# Numpy Array to store accuracy of Test Network at each iterations
test_accuracy = np.zeros(total_test_iters)

print("\nBegin Testing ...\n")
start_test = time.time()
# Running Testing, fetch accuracy
for i in range(total_test_iters):
    workspace.RunNet(test_model.net.Proto().name)
    test_accuracy[i] = workspace.FetchBlob('accuracy')
    print "Testing Iteration: #" + str(i) + " Accuracy: " + str(test_accuracy[i])
    

# Print Mean accuracy
print "\ntest_accuracy Mean:" + str( test_accuracy.mean())

end_test = time.time()

print "\nTesting: \nBatch Size= " + str(test_batch_size) + "\nNo. of Test iterations:  "+  str(total_test_iters) + "\nTime: "+ str(end_test - start_test) 

print "\nTraining: \nBatch Size= " + str(train_batch_size) + "\nNo. of Train iterations:  "+  str(total_train_iters) + "\nTime: "+ str(end_train - start_train) 

