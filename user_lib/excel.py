# -*- coding: utf-8 -*-
import xlrd
import xlwt
import pandas as pd

class Excel_Manager:
    
    def __init__(self,path):
        self.excel_path=path
        try:
            self.excel=xlrd.open_workbook(path)
        except:
            raise

    def getSheetNames(self):
        return self.excel.sheet_names()
    
    def getSheetData(self,sheet_name):
        try:
            return pd.read_excel(
                    self.excel_path,
                    sheet_name=sheet_name
                    )
        except:
            raise