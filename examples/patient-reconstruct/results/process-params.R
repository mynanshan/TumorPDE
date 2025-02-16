library(tidyverse)
library(jsonlite)
library(lubridate)
library(openxlsx)

dat = read_tsv("parameters.txt")

names(dat) = c(
  "Datetime", "Patient", "ExperimentType", "D", "rho", "x1", "x2", "x3", "t1"
)

dat = dat |> 
  arrange(desc(Datetime)) |> 
  distinct(Patient, ExperimentType, .keep_all = TRUE) |> 
  arrange(Patient)

# single scan
firstscan = dat |> 
  filter(ExperimentType=="single scan")

firstscan |> View("First-scan Results")



# two scan
dates = jsonlite::read_json("scan-dates.json")
time_diff = sapply(dates, \(subj) {
  ymd(subj[[2]]) - ymd(subj[[1]])
})

scan_dates = tibble(
  Patient=names(dates),
  ScanDate1 = sapply(dates, \(subj) {subj[[1]]}),
  ScanDate2 = sapply(dates, \(subj) {subj[[2]]})) |> 
  mutate(
    ScanDate1 = ymd(ScanDate1),
    ScanDate2 = ymd(ScanDate2),
    DaysDiff = ScanDate2 - ScanDate1
  )


multiscan <- dat |> 
  filter(ExperimentType=="multi scan") |> 
  left_join(scan_dates, by="Patient") |> 
  mutate(
    DaysTotal.est = DaysDiff / (1 - t1),
    OnsetDate.est = ScanDate2 - DaysTotal.est,
    D.scaled = D / as.numeric(DaysTotal.est),
    rho.scaled = rho / as.numeric(DaysTotal.est)
  )

multiscan |> View("Multi-scan Results")


# fixed init

dates = jsonlite::read_json("scan-dates.json")
time_diff = sapply(dates, \(subj) {
  ymd(subj[[2]]) - ymd(subj[[1]])
})

scan_dates = tibble(
  Patient=names(dates),
  ScanDate1 = sapply(dates, \(subj) {subj[[1]]}),
  ScanDate2 = sapply(dates, \(subj) {subj[[2]]})) |> 
  mutate(
    ScanDate1 = ymd(ScanDate1),
    ScanDate2 = ymd(ScanDate2),
    DaysDiff = ScanDate2 - ScanDate1
  )


fixinit <- dat |> 
  filter(ExperimentType=="fixed init") |> 
  left_join(scan_dates, by="Patient") |> 
  mutate(
    D.scaled = D / as.numeric(DaysDiff),
    rho.scaled = rho / as.numeric(DaysDiff)
  )

fixinit |> View("Fixed-init Results")

# save results
write.xlsx(list(multiscan=multiscan, firstscan=firstscan, fixinit=fixinit),
           "estimates.xlsx")



