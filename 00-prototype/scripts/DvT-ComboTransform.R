
# Tyler Edwards
# DvT Dataset Combo Transformation
# 2 - 6 - 2021
# Transforms the combos in the action data into single elements


library("readxl")
library("xlsx")
DvT = read_excel("../../DvT.xlsx", sheet = 3)

# ------------------------ P1 Combos ------------------------------------------
P1Combos = c()
P1ActCount = c()
P1Starter = c()
P1Ender = c()
for (i in 1:nrow(DvT))
{
  if(is.na(DvT$P1_RESULT[i])){} # Skip NAs
  else if(DvT$P1_RESULT[i] == "hit" 
          | toString(DvT$P1_ACTION[i]) == "throw" 
          & toString(DvT$P1_RESULT[i]) == "success") # First row is read as hit
    # Also accounts for Urien throw combos when we copy+paste
  {
    j = 1 # Combo counter and iteration
    combo = c(DvT$P1_ACTION[i])
    if(toString(DvT$P1_RESULT[i + j]) == "combo" || toString(DvT$P1_DESC[i + j]) == "combo") # Next action isn't NA
    {
      while(toString(DvT$P1_RESULT[i + j]) == "combo" || toString(DvT$P1_DESC[i + j]) == "combo") # Next RESULT is a combo 
      {
        combo = c(combo, DvT$P1_ACTION[i + j])
        j = j + 1
        # print(toString(DvT$P1_RESULT[i + j]))
        # print(toString(DvT$P1_DESC[i + j]))
      } 
      # print(c("STARTER", combo[1]))
      print(paste(combo, collapse = " -> "))
      # print(c("ENDER", tail(combo, 1)))
      
      P1Starter = c(P1Starter, combo[1]) # Record first hit as starter
      P1Ender = c(P1Ender, tail(combo, 1)) # Recorse last hit as ender
      P1Combos = rbind(P1Combos, paste(combo, collapse = " -> "))
      P1ActCount = rbind(P1ActCount, j)
    }
  }
}

# print(P1Starter)
# print(P1Ender)
P1Combos = as.data.frame(cbind(P1Combos, P1ActCount, P1Starter, P1Ender))
colnames(P1Combos) = c("Combo", "Actions", "Starter", "Ender")
P1Combos$Actions = as.numeric(P1Combos$Actions)

P1.Starter.Count = as.data.frame(table(P1Combos$Starter))
colnames(P1.Starter.Count) = c("Starter", "Frequency")
P1.Ender.Count = as.data.frame(table(P1Combos$Ender))
colnames(P1.Ender.Count) = c("Ender", "Frequency")

P1.Combo.Count = as.data.frame(table(P1Combos$Combo))
colnames(P1.Combo.Count) = c("Combo", "Frequency")
P1Combos = merge(P1.Combo.Count, P1Combos, by = "Combo")
P1Combos = unique(P1Combos)


write.xlsx2(P1Combos, "DvT-Combos.xlsx", sheetName="P1-Combos",
            col.names=TRUE, row.names=FALSE, append=TRUE)
write.xlsx2(P1.Starter.Count, "DvT-Combos.xlsx", sheetName="P1-Starters",
            col.names=TRUE, row.names=FALSE, append=TRUE)
write.xlsx2(P1.Ender.Count, "DvT-Combos.xlsx", sheetName="P1-Enders",
            col.names=TRUE, row.names=FALSE, append=TRUE)



# ------------------------ P2 Combos ------------------------------------------
P2Combos = c()
P2ActCount = c()
P2Starter = c()
P2Ender = c()
for (i in 1:nrow(DvT))
{
  if(is.na(DvT$P2_RESULT[i])){} # Skip NAs
  else if(DvT$P2_RESULT[i] == "hit" 
          | toString(DvT$P2_ACTION[i]) == "throw" 
          & toString(DvT$P2_RESULT[i]) == "success") # First row is read as hit
    # Also accounts for Urien throw combos when we copy+paste
  {
    j = 1 # Combo counter and iteration
    combo = c(DvT$P2_ACTION[i])
    if(toString(DvT$P2_RESULT[i + j]) == "combo" || toString(DvT$P2_DESC[i + j]) == "combo") # Next action isn't NA
    {
      while(toString(DvT$P2_RESULT[i + j]) == "combo" || toString(DvT$P2_DESC[i + j]) == "combo") # Next RESULT is a combo 
      {
        combo = c(combo, DvT$P2_ACTION[i + j])
        j = j + 1
        # print(toString(DvT$P2_RESULT[i + j]))
        # print(toString(DvT$P2_DESC[i + j]))
      } 
      # print(c("STARTER", combo[1]))
      print(paste(combo, collapse = " -> "))
      # print(c("ENDER", tail(combo, 1)))
      
      P2Starter = c(P2Starter, combo[1]) # Record first hit as starter
      P2Ender = c(P2Ender, tail(combo, 1)) # Recorse last hit as ender
      P2Combos = rbind(P2Combos, paste(combo, collapse = " -> "))
      P2ActCount = rbind(P2ActCount, j)
    }
  }
}

# print(P2Starter)
# print(P2Ender)
P2Combos = as.data.frame(cbind(P2Combos, P2ActCount, P2Starter, P2Ender))
colnames(P2Combos) = c("Combo", "Actions", "Starter", "Ender")
P2Combos$Actions = as.numeric(P2Combos$Actions)

P2.Starter.Count = as.data.frame(table(P2Combos$Starter))
colnames(P2.Starter.Count) = c("Starter", "Frequency")
P2.Ender.Count = as.data.frame(table(P2Combos$Ender))
colnames(P2.Ender.Count) = c("Ender", "Frequency")

P2.Combo.Count = as.data.frame(table(P2Combos$Combo))
colnames(P2.Combo.Count) = c("Combo", "Frequency")
P2Combos = merge(P2.Combo.Count, P2Combos, by = "Combo")
P2Combos = unique(P2Combos)


write.xlsx2(P2Combos, "DvT-Combos.xlsx", sheetName="P2-Combos",
            col.names=TRUE, row.names=FALSE, append=TRUE)
write.xlsx2(P2.Starter.Count, "DvT-Combos.xlsx", sheetName="P2-Starters",
            col.names=TRUE, row.names=FALSE, append=TRUE)
write.xlsx2(P2.Ender.Count, "DvT-Combos.xlsx", sheetName="P2-Enders",
            col.names=TRUE, row.names=FALSE, append=TRUE)
