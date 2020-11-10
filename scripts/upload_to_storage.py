from google.cloud import storage

list_blobs = False

client = storage.Client.from_service_account_json("TDT4173 Deep Learning Project-91d3b469375c.json")

bucket_name = "tdt4173-datasets"
bucket = client.get_bucket(bucket_name)

if list_blobs:
    blobs = bucket.list_blobs()
    for blob in blobs:
        print(blob.name)

blob_name = f"celeba/tensors/celebalign_processed_64px_100000_horizontal.torch"
source_file_name = "/Users/larsankile/GitLocal/face-recognizer-9000/celebalign_processed_64_100000_horizontal.torch"

blob = bucket.blob(blob_name)
blob.upload_from_filename(source_file_name)