# in this file we handle error 
# this will be comman code for Exception

import sys
import os




def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info() 
    file_name=exc_tb.tb_frame.f_code.co_filename
    #this fun gives a each evrey details of error  lline no, type of error
    error_message="error occured in pyton scripts name[{0}],line number[{1}],and error message[{2}] ".format(file_name,exc_tb.tb_lineno,str(error))
                                                                                                             
                                                                                                             
    return error_message                                                                                                       
    
    
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)


    
    def __str__(self):
        return self.error_message   
    