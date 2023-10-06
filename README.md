# Team 38: Shoe Pricing Model

## Introduction & Background

Reselling of consumer products is an industry that has become popular in recent years. Technology and the advent of e-commerce have created a new and accessible platform for resellers to acquire stock and find buyers efficiently and cost-effectively. We believe, however, that there is more that can be done to improve this market by utilizing machine learning techniques applied specifically to the problem of footwear. We aim to develop a model capable of estimating a shoe’s retail price based on an image. This initiative addresses a critical need in the market, where pricing decisions are often complex and time-consuming. We seek to create a proof of concept for a tool that can automatically assign shoes a price based solely on their visual features. This technology has the potential to streamline pricing strategies, optimize inventory management, and enhance the competitiveness of businesses in the footwear market. We hope that if our model proves successful, it may be possible to provide customers with a more seamless and informed shopping experience, ensuring they receive fair and accurate pricing for the shoes they desire.

## Potential Methods

Our idea is to go from image to price, which is a continuous value, which makes our project an image regression problem [4]. The approach we thought of was implementing a pre-trained CNN model and adding it to our dataset of shoes in order to perform regression based on images. Our dataset would contain the image of the shoe and the associated price label. We plan on acquiring this dataset through shoe picture data accumulated by other users on Kaggle, which includes the shoe picture (sideways to the right), price, category, brand, and more features. However, our focus would only be on the image and the price label. We plan to extend this dataset by scraping shoe data off popular sites such as Nike.com, Adidas.com, and Zappos.com. We will try out several pre-trained models, such as VGG16, VGG19, and ResNet, in order to perform image-to-price regression [2][3]. Overall, this should provide us with a baseline starting point and consider other methods or models if these pre-trained networks do not work.

## Potential Results and Discussion

Because we are creating an image regression model, we can use a variety of evaluation metrics to determine the validity of the model. First and foremost, we can look at things like the Root Mean Square Error, R^2, and Mean Absolute Error [1]. We can also conduct K-Fold cross-validation by splitting our data and keeping track of performance metrics to ensure we are not overfitting or underfitting [1]. Finally, we can also apply explorative data analysis via a box plot to be able to better understand the descriptive statistics of our results [1].


## Contribution Table
- Brandon Harris: Report - Timeline & Contribution Table            
- Emilio Aponte: Report - Introduction & Background                
- Samrat Sahoo: Video & Slides                                    
- Tawsif Kamal: Report - Potential Methods                        
- Victor Guyard: Report - Potential Results / Discussion & Sources 

## Timeline

- Week 1: We will spend this week doing some initial data collection - this is both through Kaggle as well as some initial web scraping to gather all the images.
- Week 2: We will spend this week preparing the data - this means cleaning the data (i.e., ensuring all our images are shoes facing the right side), preprocessing it as necessary (i.e., applying any necessary preprocessing or augmentation steps to ensure consistent data with the models), and organizing the dataset (mapping images to prices).
- Week 3: We will spend this week selecting some pre-trained CNNs like the ones mentioned above and ensuring we are able to run the models on our data. We are essentially setting up the data and machine learning model pipelines this week.
- Week 4: We will spend this week training and finetuning our models by optimizing the hyperparameters. We will also start writing the midpoint report for this project.
- Week 5: We will spend this week evaluating the models with the aforementioned metrics in the report. 
- Week 6: After going through the results, we will decide whether further improvement is necessary. If so, we will continue to determine weak points of our models or methodology to optimize our model’s performance further.
- Week 7: We will create a proof of concept CLI-based or UI tool to enable people to utilize our trained model in an easy manner.
- Week 8: We will write up our results in a final report (as outlined in the syllabus for this class)
- Week 9: We will complete our write-up and submit this final report. 

Responsibilties: Most responsibilties will be shared with each other but each responsibility has an "owner" who is responsibile for ensuring that task gets done. The owners are as follows:
- Brandon: Midterm & Final Reports, Finetuning Model Hyperparameters
- Emilio: Midterm & Final Reports, Evaluating Models
- Samrat: Midterm & Final Reports, Creating CLI/UI Tool for Model Inference 
- Tawsif: Midterm & Final Reports, Creating Image to Regression Data and Model Pipelines
- Victor: Midterm & Final Reports, Webscraping Data

## Potential Datasets
- https://www.kaggle.com/datasets/datafiniti/womens-shoes-prices
- https://www.kaggle.com/datasets/thedevastator/analyzing-trending-men-s-shoe-prices-with-datafi
- Webscraping from shoe shopping websites

## Works Cited

[1] Cenita, Jonelle Angelo S., et al. “Performance Evaluation of Regression Models in Predicting the Cost of Medical Insurance.” International Journal of Computing Sciences Research, vol. 7, Jan. 2023, pp. 2052–65. arXiv.org, [https://doi.org/10.25147/ijcsr.2017.001.1.146](https://doi.org/10.25147/ijcsr.2017.001.1.146).

[2] Hou, Sujuan, et al. Deep Learning for Logo Detection: A Survey. arXiv, 9 Oct. 2022. arXiv.org, [https://doi.org/10.48550/arXiv.2210.04399](https://doi.org/10.48550/arXiv.2210.04399).

[3] Oliveira, João, et al. ‘Footwear Segmentation and Recommendation Supported by Deep Learning: An Exploratory Proposal’. Procedia Computer Science, vol. 219, 2023, pp. 724–735, [https://doi.org10.1016/j.procs.2023.01.345](https://doi.org10.1016/j.procs.2023.01.345).

[4] Yang, Richard R., et al. AI Blue Book: Vehicle Price Prediction Using Visual Features. arXiv, 18 Oct. 2018. arXiv.org, [https://doi.org/10.48550/arXiv.1803.11227](https://doi.org/10.48550/arXiv.1803.11227).

## Video & Presentation

- [Video](https://youtu.be/ipoQmIiKBfM)
- [Presentation](https://docs.google.com/presentation/d/1QrepxgSp1oowxFkXETYxBXCTLJOTINg0yjObIau99ds/edit?usp=sharing)