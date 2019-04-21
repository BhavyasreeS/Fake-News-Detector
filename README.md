# what's working
1. Run svm1.py
     It calles getembeddeings for preprocessing, which splits the data into training and testing. Now svm1.py builds the model using svm and model is saved as savedsvm.sav. 
2. Run svm2.py
     Only the test set is loaded from yte.npy and xte.npy. The saved model is loaded and test data is predicted. Confusion matrics is then plotted.

Result: Works fine. Accuracy and confusion matrix obtained correctly. Now, training and testing are in separate files. The trained model is saved for later use.


#Changes needed
1. getembeddings processes the entire data from train.csv. It splits the data into training and testing(80% for training and 20% for testing)When a single instance is given this splitting must be avoided and must be converted to vector.???? This single instance must be converted to a .npy file for testing.
2. Once .npy file is created prediction can be made using the previously saved model.

Changes made.
        In getembeddings file I tried making the train size and test size as 1 as there is only a single instance. However confusion still prevails. In later steps I discarded the train set and focused on the test set which is renamed as xte1 and yte1. However this approach doesn't seem to work. Codes towards the end of getembeddings() in the file getembeddings.py not clearly understood.
        
Errors encountered: changes made to the getembeddings file(renamed as getembeddingstest din'd work as expected). Value predicted always remains 1. 
