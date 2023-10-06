# ML-Team-38

## Introduction & Background

Reselling of consumer products is an industry that has become popular in recent years. Technology and the advent of e-commerce have created a new and accessible platform for resellers to acquire stock and find buyers efficiently and cost-effectively. We believe, however, that there is more that can be done to improve this market by utilizing machine learning techniques applied specifically to the problem of footwear. We aim to develop a classification model capable of estimating a shoe’s retail price based on an image. This initiative addresses a critical need in the market, where pricing decisions are often complex and time-consuming. We seek to create a proof of concept for a tool that can automatically assign shoes to different price categories, such as budget, mid-range, or luxury, based solely on their visual features. This technology has the potential to streamline pricing strategies, optimize inventory management, and enhance the competitiveness of businesses in the footwear market. We hope that if our classifier proves successful, it may be possible to provide customers with a more seamless and informed shopping experience, ensuring they receive fair and accurate pricing for the shoes they desire.

## Potential Methods

Our idea is to go from image to price, which is a continuous value. The approach we thought of was implementing a pre-trained CNN model and adding it to our dataset of shoes in order to perform regression based on images. Our dataset would contain the image of the shoe and the associated price label. We plan on acquiring this dataset through shoe picture data accumulated by other users on Kaggle, which includes the shoe picture (sideways to the right), price, category, brand, and more features. However, our focus would only be on the image and the price label. We plan to extend this dataset by scraping shoe data off popular sites such as Nike.com, Addidas.com, and Zappos.com. We will try out several pre-trained models, such as VGG16, VGG19, and ResNet, in order to perform image-to-price regression. Overall, this should provide us with a baseline starting point and consider other methods or models if these pre-trained networks do not work.

## Potential Results and Discussion

## Contribution Table
| **Person**     | **Contribution**                                  |
| -------------- | ------------------------------------------------- |
| Brandon Harris | Report - Timeline & Contribution Table            |
| Emilio Ayonte  | Report - Introduction & Background                |
| Samrat Sahoo   | Video & Slides                                    |
| Tawsif Kamal   | Report - Potential Methods                        |
| Victor Guyard  | Report - Potential Results / Discussion & Sources |

## Timeline

## Works Cited

[1] Hou, Sujuan, et al. Deep Learning for Logo Detection: A Survey. arXiv, 9 Oct. 2022. arXiv.org, [https://doi.org/10.48550/arXiv.2210.04399](https://doi.org/10.48550/arXiv.2210.04399).

[2] Oliveira, João, et al. ‘Footwear Segmentation and Recommendation Supported by Deep Learning: An Exploratory Proposal’. Procedia Computer Science, vol. 219, 2023, pp. 724–735, [https://doi.org10.1016/j.procs.2023.01.345](https://doi.org10.1016/j.procs.2023.01.345).

[3] Yang, Richard R., et al. AI Blue Book: Vehicle Price Prediction Using Visual Features. arXiv, 18 Oct. 2018. arXiv.org, [https://doi.org/10.48550/arXiv.1803.11227](https://doi.org/10.48550/arXiv.1803.11227).