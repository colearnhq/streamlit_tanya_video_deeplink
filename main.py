import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import boto3
from io import StringIO
import logging
import collections
import datetime
from stqdm import stqdm

class TailLogHandler(logging.Handler):

    def __init__(self, log_queue):
        logging.Handler.__init__(self)
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.append(self.format(record))


class TailLogger(object):

    def __init__(self, maxlen):
        self._log_queue = collections.deque(maxlen=maxlen)
        self._log_handler = TailLogHandler(self._log_queue)

    def contents(self):
        return '\n'.join(self._log_queue)

    @property
    def log_handler(self):
        return self._log_handler

@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

client = boto3.client('s3',
                    aws_access_key_id = st.secrets["aws_access_key_id"],
                    aws_secret_access_key = st.secrets["aws_secret_access_key"]
                    )

bucket = st.secrets["bucket"]
prefix = st.secrets["prefix"]

tag_list = ['Question ID', 'Grade (Stream)', 'Chapter', 'Difficulty level', 
        'Label 1', 'Label 2', 'Label 3', 'Skill', 'Skill ID']

headers = {
  'Authorization': st.secrets["Authorization"],
  'Content-Type': st.secrets["Content-Type"]
}

st.set_page_config(page_title="Tanya Deeplink", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Video Deep Link Generator")
# === PROCESSING SECTION ===
st.header("File Processing")
st.markdown('''
File must contain these columns:
* One of **Question ID** or **UUID**
* Campaign
* Channel
* Feature
* Subject
* Grade (Stream)
* Chapter
* Difficulty level
* Label 1
* Label 2
* Label 3
* Skill
* Skill ID

Subjects should be:
1. Maths
2. Physics
3. Chemistry
''')
input_csv = st.file_uploader("Choose File", type="csv", accept_multiple_files=False, key=None, help=None)

if input_csv != None:
    video_file = pd.read_csv(input_csv)

    if ("Question ID" in video_file.columns) or ("UUID" in video_file.columns):        
        if st.button('Process File'):
            
            # logging initialization
            logger = logging.getLogger("__process__") # creating logging variable
            logger.setLevel(logging.DEBUG) # set the minimun level of loggin to DEBUG
            formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s") # logging format that will appear in the log file
            tail = TailLogger(10000) # amount of log that is saved
            log_handler = tail.log_handler # variable log handler
            log_handler.setFormatter(formatter) # set formatter to handler
            logger.addHandler(log_handler) # adding file handler to logger
            
            try:
                with st.spinner('Loading Files'):
                    destination_bucket = st.secrets["destination_bucket"]
                    destination_prefix = st.secrets["destination_prefix"]
                    destination_file_name = 'doubts.csv'
                    destination_file = destination_prefix + destination_file_name
                    obj = client.get_object(Bucket= destination_bucket, Key= destination_file) 
                    doubts_csv = pd.read_csv(obj['Body'])
                    # doubts_csv = pd.read_csv('test_files/doubts.csv')
                    
                    file_name = 'processed_file_id.csv'
                    file = prefix + file_name
                    obj = client.get_object(Bucket=bucket, Key=file)
                    previous_file = pd.read_csv(obj['Body'])  # 'Body' is a key word

                    all_file_ids = list(previous_file['file_id'].values)
                    last_file_id = all_file_ids[-1]
                    curr_id = last_file_id+1

                data = doubts_csv[['questionId', 'question_uuid']]
                data.columns = ['Question ID', 'UUID']

                if "Question ID" in video_file.columns:
                    qid_df = video_file[~video_file['Question ID'].isna()].copy()
                    try: 
                        qid_df = qid_df.drop(columns=['UUID'])
                    except:
                        print("Only-Contain Question ID")
                        uuid_df = pd.DataFrame()
                    qid_df = pd.merge(qid_df, data, how='left', on='Question ID')

                if "UUID" in video_file.columns:
                    uuid_df = video_file[~video_file['UUID'].isna()].copy()
                    try:
                        uuid_df = uuid_df.drop(columns=['Question ID'])
                    except:
                        print("Only-Contain UUID")
                        qid_df = pd.DataFrame()
                    uuid_df = pd.merge(uuid_df, data, how='left', on='UUID')

                video_file = pd.concat([qid_df, uuid_df]).reset_index(drop=True)

                status = []
                deep_link_url = []

                start_time = datetime.datetime.now()
                logger.info("Deeplink Processing Start")

                for i in stqdm(video_file.index):
                    uuid = video_file.iloc[i]['UUID']
                    campaign = video_file.iloc[i]['Campaign']
                    channel = video_file.iloc[i]['Channel']
                    feature = video_file.iloc[i]['Feature']
                    subject = video_file.iloc[i]['Subject']
                    tags = video_file.iloc[i][tag_list].values
                    tags = tags[~pd.isnull(tags)]
                    tags = [str(i) for i in list(tags)]

                    data = {
                        "campaign": campaign,
                        "channel": channel,
                        "feature": feature,
                        "subject": subject,
                        "tags": tags
                    }

                    payload = json.dumps(data)
                    url = "https://api.colearn.id/ads/v3/doubts/"+str(uuid)+"/share/"
                    # print(url)
                    tries = 3
                    for o in range(tries):
                        print(f"{i} - {o}")
                        logger.debug(f"{i} - {o}")
                        try:
                            response = requests.request("POST", url, headers=headers, data=payload)
                            # print(response.json()['data']['deep_link_url'])
                            status.append(response.json()['status'])
                            deep_link_url.append(response.json()['data']['deep_link_url'])
                            break
                        except Exception as e:
                            if o == tries-1:
                                status.append(None)
                                deep_link_url.append(None)
                            logger.error(f"{i} Error Requesting : {e}")
                            continue
                
                video_file['status'] = status
                video_file['deep_link_url'] = deep_link_url
                
                time_taken = datetime.datetime.now() - start_time
                
                all_file_ids.append(curr_id)
                current_file = pd.DataFrame(all_file_ids, columns=['file_id'])

                output_file_names = ['streamlit_deeplink_'+str(curr_id)+'.csv', 'processed_file_id.csv']
                output_files = [video_file, current_file]

                for i in range(2):
                    try:
                        with StringIO() as csv_buffer:
                            output_files[i].to_csv(csv_buffer, index=False)
                            output_file = prefix + output_file_names[i]
                            print(output_file)
                            response = client.put_object(Bucket=bucket, Key=output_file, Body=csv_buffer.getvalue())
                    except Exception as e:
                        print(e)
                        logger.error(f"Error Uploading Files : {e}")
                
                st.text(f"Processed File : {video_file.shape[0]}")
                st.text(f"Error File : {len(video_file[video_file['status'].isna()])}")
                st.text(f"Time taken : {time_taken}")
                st.text(f"File ID = {curr_id}")

                logger.debug(f"Processed File : {video_file.shape[0]}")
                logger.debug(f"Error File : {len(video_file[video_file['status'].isna()])}")
                logger.debug(f"Time taken : {time_taken}")
                logger.debug(f"File ID = {curr_id}")

            except Exception as e:
                logger.error(f"Error Processing Files : {e}")
                st.error(f'Error Processing File : {e}')
            
            val_log = tail.contents() # extracting the log 

            # deleting all loggin variable for the current process
            log_handler.close()
            logging.shutdown()
            logger.removeHandler(log_handler)
            del logger, log_handler

            # saving the log file to S3
            try:
                log_filename = f"deeplink_process_log_{curr_id}.txt" # the name of the log file
                client.put_object(Bucket=bucket, Key=prefix + log_filename, Body=val_log)
                print(prefix + log_filename)
                print(val_log)
            except Exception as e:
                print(e)

    else:
        st.error("CSV File does not have a \"Question ID\" and \"UUID\" Column")

# === DOWNLOAD SECTION ===
st.header("")
st.header("")
st.header("Download Processed File")
dwn_file_id = st.text_input("Type in file id to download deeplink processed file")

if dwn_file_id != "":
    dwn_file_name = 'streamlit_deeplink_'+str(dwn_file_id)+'.csv'
    dwn_file = prefix + dwn_file_name

    try:
        obj = client.get_object(Bucket= bucket, Key= dwn_file)
        dwn_data = pd.read_csv(obj['Body']) # 'Body' is a key word
        csv = convert_df(dwn_data)

        st.download_button(
            label="Download Result",
            data=csv,
            file_name='deeplink_result'+str(dwn_file_id)+'.csv',
            mime='text/csv',
        )
    
    except:
        st.error("File ID not found in S3")