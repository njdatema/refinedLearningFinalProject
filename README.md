# refinedLearningFinalProject

Final project for the CS5841/EE5841 Spring 2026 semester

This includes an assortement of tests comparing a image processing techniques 
with current state of the art crowd counting machine learning models. 

You may find all code in /notebooksAndCode which include some jupiter notebooks 
and some python. You may also find the proposal and previous versions of the final report 
manuscript in /supportingDocuments

Directory
        /notebooksAndCode- 
            /Base Dataset and Annotations-
                Includes all images, annotations, and processes of how we annotated and the program that enabled us to have a "attention zone" to guage people specifically watching
            /CSRNet_model-
                All files used in the training and transfer learning process for CSRNet
            /P2PNet -
                All files used in the training and transfer learning process for P2PNet
            /MLP_Model_and_Supporting_Code- 
                /Annotated Data- All characteristic data used 
                    to train model. "WITHCART" means with Cartesion is simple model per paper (X,Y), Otherwise polar based (advanced MLP per paper).
                annotationBulkImageProcessing.PY- File to
                      translate our annotated images into characteristic data to train MLP
                BACKGROUND2.png- spliced background image that
                      we preform subtraction with input
                ModelMLPcartesiuan.ipynb - Notebook containing 
                      simple MLP model per paper  (x,y)
                ModelMLPpolar.ipynb - Notebook containing 
                      advanced MLP model per paper  (polar coordiantes based on displacement from center camera)
                      
        /supportingDocuments-
            Contains proposal, progress report, and final presentation slides
