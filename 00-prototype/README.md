# 00 - Proof of Concept
## Using manually collected data to explore its potential applications.

### [_**More details in the article.**_](https://medium.com/@TyEdwardz/raging-download-a3ff6fcc42cb)

![Normals Comparison](https://user-images.githubusercontent.com/69095276/114776325-58e80300-9d40-11eb-9f6c-7d477ebcc63b.png)

### Data
_Data was collected from a video of an online Street Fighter 5 matches between top Japanese players Daigo Umehara and Tokido. The video has since been deleted, 
but the data was collected by pausing the video anytime a player performed an action and manually recording it into the spreadsheet. (Basic movements like walking and crouching were excluded.)_

**Dataset** | **Description** |
--- | --- | 
DvT.xlsx | Master dataset that cleans up DvT(Raw).xlsx and adds sheets showing who won each game/round| 
DvT(Raw).xlsx | Raw copy of the manually recorded data from the video | 
DvT-Combos.xlsx | Output from DvT-ComboTransform.R that shows the frequency of combos used in the match.  | 
DvT-Turns.xlsx |  Output from DvT-TurnTransform.R that shows the frequency of "turns" lost/won after actions and also block strings. | 
BoomDamage.xlsx | Output from RoundAnalysis.R thats shows the amount of damage taken from Guile's (Daigo) *Sonic Boom* in each round. | 

### Scripts
_R scripts used to clean and aggregate data for visualizations_
**Script** | **Description** |
--- | --- | 
DvT-ComboTransoform.R | Transforms the combos in the action data into single elements|
DvT-RoundAnalysis.R |  Formats data into actions per round and runs t-test. Also calculates average damage of sonicbooms per round|
DvT-TurnTransoform.R | Identifies turns gained and lost after blocks in the action data|


### Visulizations
_Tableau visulizations created to easily analyze some of the data_
**Visuals** | **Description** |
--- | --- | 
DvT - Action + Result Visuals.pdf | Graphs created from DvT.xlsx, displaying the most used actions from each player|
DvT - Combo Visuals.pdf | Graphs created from DvT-Combos.xlsx, displaying the most used combos and combo started each player landed|
DvT - Round Visuals.pdf | Graphs comparing the amount of actions each player performed each round |
DvT – Turn Visualizations.pdf | Graphs created from DvT-Turns.xlsx, displaying how many times each player won/lost a turn after specific actions or block strings (Winning a "turn" means sucssfully performing an action/attack after making your opponent block something, losing a turn means you were forced to block or were hit afterwards)|
