# Dogs Image Classification using pre-trained  ResNet18  CNN model  in AWS SageMake

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
Using AWS Sagemaker to finetune a pretrained model that can perform image classification. You will have to use Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices to finish this project.
Will be using the dog breed classification dataset "https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip" , to classify between 133 different breeds of dogs.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

-Choosed a pre-trained Resnet18 model and used transfer learning to train the model on (dog breed classification) dataset.
-Optimized for the learning rate, batch size and the number of epochs to use.

    For the learning rate, I chose a linear search space from 0.001 to 0.1.
    For the batch size, I chose a categorical search space of (16, 32, 64)

Completed training jobs.
![Training jobs.png](framework images/Training jobs.png)

Completed Hyperparameter tuning job.
![Hyperparameter tuning job.png](framework images/Hyperparameter tuning job)


Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
Performed debugging and profiling in Sagemaker using, Sagemaker Debugger and Profiler respectively. To perform debugging using Sagemaker Debugger, used the following steps.

    Added hooks for the debugger and the profiler in the train() and test() functions and set them to their respective modes.
    In the main() function created the hook and registered the model to the hook. This hook is passed to the train() and test() functions.
    In the notebook, configured the debugger rules and the hook parameters.
    Created profiler rules and config.

![graph.png](framework images/graph.png)

### Results

The validation loss first decreases then increases as the steps increases which means we can applay early stopping to fix that. We can also use Debugger to enable auto-termination, which stops the training when a rule triggers. For our use case, doing so reduces compute time by more than half (orange curve)

## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

The model was deployed to an endpoint on Sagemaker with an instance type of "ml.m5.large". The following image shows the deployed endpoint in Sagemaker.

Running Endpoint

![Endpoint.png](framework images/Endpoint.png)

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
