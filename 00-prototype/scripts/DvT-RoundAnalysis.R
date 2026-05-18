
# Tyler Edwards
# DvT Dataset Turn Transformation
# 4 - 7 - 2021
# Formats data into actions per round and runs t-t\test
# Also calculates average damage of sonicbooms per round

library("readxl")
library("xlsx")

DvT = read_excel("../Datasets/DvT.xlsx", sheet = 3)
DvT.Rounds = read_excel("../Datasets/DvT.xlsx", sheet = 2)
DvT.Rounds = DvT.Rounds[order(DvT.Rounds$MATCH_ID),]

# Creating dataset of count of P1 Actions by Match and Round
P1.ActPerRound = DvT[!is.na(DvT$P1_ACTION),]
P1.ActPerRound = as.data.frame(table(P1.ActPerRound$MATCH_ID, P1.ActPerRound$ROUND))
P1.ActPerRound = P1.ActPerRound[P1.ActPerRound$Freq > 0, ]
P1.ActPerRound = P1.ActPerRound[order(P1.ActPerRound$Var1), ]
P1.ActPerRound$Result = DvT.Rounds$RESULT
colnames(P1.ActPerRound) = c("Match", "Round", "Actions", "RoundResult")

# Split dataframe into Wins and Losses and t-test
P1Act.Wins = P1.ActPerRound[P1.ActPerRound$RoundResult == 0,]
P1Act.Loss = P1.ActPerRound[P1.ActPerRound$RoundResult == 1,]
t.test(P1Act.Wins$Actions, P1Act.Loss$Actions)
# p-value = 0.2883
# Mean Actions (Wins) = 52.30
# Mean Actions (Losses) = 58.25

# Same thing but P2
P2.ActPerRound = DvT[!is.na(DvT$P2_ACTION),]
P2.ActPerRound = as.data.frame(table(P2.ActPerRound$MATCH_ID, P2.ActPerRound$ROUND))
P2.ActPerRound = P2.ActPerRound[P2.ActPerRound$Freq > 0, ]
P2.ActPerRound = P2.ActPerRound[order(P2.ActPerRound$Var1), ]
P2.ActPerRound$Result = DvT.Rounds$RESULT
colnames(P2.ActPerRound) = c("Match", "Round", "Actions", "RoundResult")

P2Act.Wins = P2.ActPerRound[P2.ActPerRound$RoundResult == 1,]
P2Act.Loss = P2.ActPerRound[P2.ActPerRound$RoundResult == 0,]
t.test(P2Act.Wins$Actions, P2Act.Loss$Actions)
# p-value = 0.00001438
# Mean Actions (Wins) = 52.08333
# Mean Actions (Losses) = 37.5

# Merging
cor(P1.ActPerRound$Actions, P2.ActPerRound$Actions)
# R = 0.5991

# Calculation Percent difference in actions to account for longer rounds
PDif = (P1.ActPerRound$Actions - P2.ActPerRound$Actions) / P1.ActPerRound$Actions
PDif = as.data.frame(cbind(PDif, P1.ActPerRound$RoundResult))
t.test(PDif$PDif[PDif$V2 == 0], PDif$PDif[PDif$V2 == 1])
# p-value = 0.004941
# Mean (P1 Win) = 0.2803
# Mean (P2 Win) = 0.0304

# ------------------------------------------------------------------------
# Sonicboom Analysis

Booms = DvT[!is.na(DvT$P1_ACTION),] # Dataframe of only P1 Actions
# Filtering for all projectiles
Booms = Booms[Booms$P1_ACTION == "sonicboom" | 
              Booms$P1_ACTION == "ex-sonicboom" |
              Booms$P1_ACTION == "vs2" |
              Booms$P1_ACTION == "vt1",]
table(Booms$P1_ACTION)

# Filter booms used in combos or that didn't hit
Booms = Booms[!Booms$P1_RESULT == "combo" & 
              !Booms$P1_RESULT == "(counterhit)" &
              !Booms$P1_RESULT == "whiff" &
              !Booms$P1_RESULT == "destroyed",]
table(Booms$P1_RESULT)

# No need to be specific with hits and blocks
Booms$P1_RESULT[Booms$P1_RESULT == "c.block" | Booms$P1_RESULT == "s.block"] = "block"
Booms$P1_RESULT[Booms$P1_RESULT == "hit" | Booms$P1_RESULT == "trade"] = "hit"
table(Booms$P1_RESULT)

# Adding damage column
Booms$Damage = 999999

Booms$Damage = ifelse(Booms$P1_ACTION == "sonicboom" & Booms$P1_RESULT == "hit", 60,
               ifelse(Booms$P1_ACTION == "sonicboom" & Booms$P1_RESULT == "block", 10,
               ifelse(Booms$P1_ACTION == "ex-sonicboom" & Booms$P1_RESULT == "hit", 100,
               ifelse(Booms$P1_ACTION == "ex-sonicboom" & Booms$P1_RESULT == "block", 18,
               ifelse((Booms$P1_ACTION == "vs2" | Booms$P1_ACTION == "vt1") & Booms$P1_RESULT == "hit", 40,
               ifelse((Booms$P1_ACTION == "vs2" | Booms$P1_ACTION == "vt1") & Booms$P1_RESULT == "block", 7, 99999))))))
                             

Booms$ROUND = as.factor(Booms$ROUND)
Booms$Damage = as.numeric(Booms$Damage)

# Calculating total damage booms caused per round
BoomsPerRound = aggregate(x = Booms$Damage,
          by = list(Booms$MATCH_ID, Booms$ROUND),
          FUN = sum)
colnames(BoomsPerRound) = c("Match", "Round", "Damage")
summary(BoomsPerRound)

# Calculating total chip damage from booms per round
Booms.Block = Booms[Booms$P1_RESULT == "block",]
ChipPerRound = aggregate(x = Booms.Block$Damage,
                          by = list(Booms.Block$MATCH_ID, Booms.Block$ROUND),
                          FUN = sum)
colnames(ChipPerRound) = c("Match", "Round", "Damage")
summary(ChipPerRound)

# Plotting
BoomBox = cbind(BoomsPerRound$Damage, ChipPerRound$Damage)
boxplot(BoomBox, las = 1, horizontal = FALSE, 
        names = c("Total Damage", "Chip Damage"),
        main = "Sonicboom Damage", col = "gold")

# Writing to file to create better graph in Tableau
write.csv(BoomsPerRound, "BoomDmg.csv", row.names=FALSE)
write.csv(ChipPerRound, "ChipDmg.csv", row.names=FALSE)
