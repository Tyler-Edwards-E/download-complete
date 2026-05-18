
# Tyler Edwards
# DvT Dataset Turn Transformation
# 3 - 3 - 2021
# Identifies turns gained and lost after blocks in the action data


library("readxl")
library("xlsx")
library(data.table)

DvT = read_excel("../../DvT.xlsx", sheet = 3)

# ------------------------ P1 Turns ------------------------------------------
P1Turns = c()
P1Followups = c()

for (i in 1:nrow(DvT)) # Parse rows
{
  if(is.na(DvT$P1_ACTION[i])){} # Skip NAs
  else if(toString(DvT$P1_RESULT[i]) == "c.block" 
          || toString(DvT$P1_RESULT[i]) == "s.block") # Looking for blocks instead of hit
  {
    P1Action = c(DvT$P1_ACTION[i])
    if(!is.na(DvT$P1_ACTION[i + 1])) # Next action isn't NA
    {
      if(toString(DvT$P1_RESULT[i + 1]) == "(counterhit)" 
         || toString(DvT$P1_RESULT[i + 1]) == "whiff") # Next action isn't sucessful
      {
        P1Action = c(P1Action, DvT$P1_ACTION[i+1], "Lost")
        
      }
      else
      {
        P1Action = c(P1Action, DvT$P1_ACTION[i+1], "Won")
      }
    }
    else # Player doesn't have next action
    {
      P1Action = c(P1Action, "", "Lost")
    }
    
    P1Turns = rbind(P1Turns, P1Action)
    
    P1F = c(paste(P1Action[1:2], collapse = " -> "), P1Action[3])
    P1Followups = rbind(P1Followups, P1F)
    
  }
}

P1Turn.Final = as.data.frame(as.matrix.data.frame(table(P1Turns[,1], P1Turns[,3]), rownames.force = TRUE))
P1Turn.Final$Total = P1Turn.Final$V1 + P1Turn.Final$V2
setDT(P1Turn.Final, keep.rownames=TRUE)[]
colnames(P1Turn.Final) = c("Action", "Lost", "Won", "Total")

P1Followup.Final = as.data.frame(as.matrix.data.frame(table(P1Followups[,1], P1Followups[,2]), rownames.force = TRUE))
P1Followup.Final$Total = P1Followup.Final$V1 + P1Followup.Final$V2
setDT(P1Followup.Final, keep.rownames=TRUE)[]
colnames(P1Followup.Final) = c("Action", "Lost", "Won", "Total")

# Creating files
write.xlsx2(P1Turn.Final, "DvT-Turns.xlsx", sheetName="P1-Turns",
            col.names=TRUE, row.names=FALSE, append=TRUE)
write.xlsx2(P1Followup.Final, "DvT-Turns.xlsx", sheetName="P1-Followups",
            col.names=TRUE, row.names=FALSE, append=TRUE)




# ------------------------ P2 Turns ------------------------------------------
P2Turns = c()
P2Followups = c()

for (i in 1:nrow(DvT)) # Parse rows
{
  if(is.na(DvT$P2_ACTION[i])){} # Skip NAs
  else if(toString(DvT$P2_RESULT[i]) == "c.block" 
          || toString(DvT$P2_RESULT[i]) == "s.block") # Looking for blocks instead of hit
  {
    P2Action = c(DvT$P2_ACTION[i])
    if(!is.na(DvT$P2_ACTION[i + 1])) # Next action isn't NA
    {
      if(toString(DvT$P2_RESULT[i + 1]) == "(counterhit)" 
         || toString(DvT$P2_RESULT[i + 1]) == "whiff") # Next action isn't sucessful
      {
        P2Action = c(P2Action, DvT$P2_ACTION[i+1], "Lost")
        
      }
      else
      {
        P2Action = c(P2Action, DvT$P2_ACTION[i+1], "Won")
      }
    }
    else # Player doesn't have next action
    {
      P2Action = c(P2Action, "", "Lost")
    }
    
    P2Turns = rbind(P2Turns, P2Action)
    
    P2F = c(paste(P2Action[1:2], collapse = " -> "), P2Action[3])
    P2Followups = rbind(P2Followups, P2F)
    
  }
}

P2Turn.Final = as.data.frame(as.matrix.data.frame(table(P2Turns[,1], P2Turns[,3]), rownames.force = TRUE))
P2Turn.Final$Total = P2Turn.Final$V1 + P2Turn.Final$V2
setDT(P2Turn.Final, keep.rownames=TRUE)[]
colnames(P2Turn.Final) = c("Action", "Lost", "Won", "Total")

P2Followup.Final = as.data.frame(as.matrix.data.frame(table(P2Followups[,1], P2Followups[,2]), rownames.force = TRUE))
P2Followup.Final$Total = P2Followup.Final$V1 + P2Followup.Final$V2
setDT(P2Followup.Final, keep.rownames=TRUE)[]
colnames(P2Followup.Final) = c("Action", "Lost", "Won", "Total")

# Creating files
write.xlsx2(P2Turn.Final, "DvT-Turns.xlsx", sheetName="P2-Turns",
            col.names=TRUE, row.names=FALSE, append=TRUE)
write.xlsx2(P2Followup.Final, "DvT-Turns.xlsx", sheetName="P2-Followups",
            col.names=TRUE, row.names=FALSE, append=TRUE)