# Machine Learning for Big Data

The main presentation can be found [here](https://paper.dropbox.com/doc/Machine-Learning-for-Big-Data--AzHLzwLY46w49jtiNSOb~_6ZAQ-7bJT1ol8V5VTYJzjfuBrS). There's also a markdown export in this repository, but it's not guaranteed to be up-to-date.

Instructions for setting up AWS for the last exercise are [here](https://paper.dropbox.com/doc/ML-for-Big-Data-preparation--AzFuLVt64sLM48Rd6pzSwwL3AQ-MBBnb5SpzREtEcDMdg8GY).


## List of all notebooks
- [Data size and model capacity](https://colab.research.google.com/drive/1mX9OJd7fCa2eGJN4A4xYkdE-USqYyKr0)
- [Spark Introduction](https://colab.research.google.com/drive/1gF_v3p_KyArAfiimqhpJ09ySTpDDtlzs)
- [Spark MLLib](https://colab.research.google.com/drive/1Wgz1sXd0dBpCgFRN3LmzF4Yhay--y_ji)
- [Dask & Dask-ML](https://colab.research.google.com/drive/1P19JculqNRAZgX7tsRdMAmUMlJZ4hkAS)
- [Scikit-learn incremental learning](https://colab.research.google.com/drive/1XsYH9wfTJDjMmwlpbdg5HrjFnkKJ3mft#scrollTo=94i_3y36Qe71)
- [MLLib model on EMR cluster](https://colab.research.google.com/drive/1IUBJ9fORNofsIPm31kbuOTlUYSnOjiz-)



# Machine Learning for Big Data

# Abstract

The aim of this course is to present an overview of tools and concepts from machine learning on big data. After going through the course participants should be able to tell what is the right tool to use for the given problem, whether there is a simpler solution and how to avoid common mistakes. Special attention will be given to Spark as a universal tool that can be used for both big data processing and machine learning.

----------
# Timeline
-  9:00 - 10:30 - **Overview of Big Data concepts and tools**
- 10:30 - 10:45 - Break
- 10:45 - 12:00 - **Introduction to Spark**
- 12:00 - 13:00 - Lunch
- 13:00 - 14:30 - **ML strategies for Big Data**
- 14:30 - 14:45 - Break
- 14:45 - 16:00 - **Frameworks (Spark MLlib)**
- 16:00 - 16:15  - Break
- 16:15  - ?         - **Training MLLib model in the Cloud**
----------
# About me…
- Worked as a data scientist / data engineer / ML engineer / software developer / devops for the last 10 years in startups / corporates / academy / consulting
- A *generalist*: “A person with a wide array of knowledge on a variety of subjects, useful or not”
- Loves math (probability and mathematical analysis in particular), but is pretty bad at it
- Advocate of simplicity and believer that even *a small data can go a long way* (“*I s malými daty se dá hrát velké divadlo*”)
# Overview of Big Data concepts and tools
## Are you ready for ML on big data?
![https://hackernoon.com/the-ai-hierarchy-of-needs-18f111fcc007](https://paper-attachments.dropbox.com/s_86CFF5479D5C05A018B2538409D8E30C1596B9FEC4E9972D3F9F51C0771A30C2_1586158742368_image.png)

- Are you building on strong foundations?
- Are you already getting **value from small data** or hopping right into big data?
- If you can’t make it work on small data with simple algorithms, then big data won’t save you
----------
## From small to big data and estimating its value
- Always make small steps
    1. Data quality control on small data
    2. Simple predictive model on small data (as simple as mean by categories)
    3. Verify all assumptions in AB test
    4. More complex model on small data (but still explainable, e.g. regression)
    5. Verify all assumptions again
    6. Can we translate model improvement into **added business value**?
    7. Train same model on big data and compare it to the old one
    8. Finally train and deploy complex model on big data


- Or make big steps, but then you have to assume **everything is wrong**

- Unless the scale of your company is huge,  it’s likely that you’ll come to a point where marginal improvements are not worth the effort anymore and only make it **less maintainable**

- Find bottlenecks
    - Is it data quality?
    - Is it lack of enough **informative** training data (e.g. lack of positives)?
    - Is it lack of adaptability & difficult innovation?
    - Is it performance on small data vs big data?


- Why not to rush into big data?
    - **Slower iterations** - is it better for prototyping to train 10 small models per day or 1 big model?
    - Hard to find data quality issues and debug model
    - Need infrastructure for big data processing
----------
## Bayesian Optimal Learner
![](https://paper-attachments.dropbox.com/s_86CFF5479D5C05A018B2538409D8E30C1596B9FEC4E9972D3F9F51C0771A30C2_1586163689332_image.png)

- The Bayes Optimal Learner is a probabilistic model that makes **the most probable prediction** for a new example and has the minimum possible error (Bayes error)
- **It cannot be found in practice**, but you should be aware it exists and you cannot do better because of **irreducible noise**
- If your classifier has **enough capacity** and you have **enough data**, performance will converge to the Bayes error
- Bayes error = 0 if problem is deterministic, but this is often not the case
- More data can get us closer to the Bayes error, but is it worth it?
![It’s not as much about the model as it is about the data size (note the logarithmic x-axis)](https://paper-attachments.dropbox.com/s_86CFF5479D5C05A018B2538409D8E30C1596B9FEC4E9972D3F9F51C0771A30C2_1586853538970_image.png)

----------
## Row vs column-oriented database
![](https://assets-global.website-files.com/5debb9b4f88fbc3f702d579e/5e08f341edb7545ceaa16494_672340c374e04c44b8d01a085a93ad5f.png)


Both typically use SQL interface to fetch data, but each is optimized for a different types of queries by storing data differently on disk / in memory.

**Row oriented database**

- best for transactional queries (reads, writes, updates, …)
- uses **indexing** to efficiently search for several records by their ID
- designed to store **normalized data**
- example of an efficient query
    SELECT * FROM sales
    JOIN product_types USING type_id
    WHERE timestamp between '2020-01-01' and '2020-01-02' and product_type.name = 'toy'
- example of an "inefficient” query
    SELECT sum(profit) FROM sales

**Column oriented database**

- best for analytical queries using as few columns as possible
- usually **doesn’t support indexing** (there are alternatives like partitioning, hash key, distributed key)
- designed to store **denormalized data**
- efficient compression
- example of an efficient query
    SELECT month, sum(profit) FROM sales
    GROUP BY 1
- example of an inefficient query
    SELECT * FROM sales WHERE id = 123

**Practial tip**
Don’t rush into column-oriented databases. PostgreSQL can store terabytes of data and is universal. It’s often easier to sample or aggregate data in an ETL job rather than having to manage another database.

----------
## HDFS (Hadoop Distributed File System)
- Special **filesystem** for distributed data that supports hierarchical access with **fault detection and recovery**
- **Write-once-read-many** means data cannot be updated, only appeneded to or truncated (or replaced)
- Moving computation is cheaper than moving data
- Saves data into **blocks** with a typical size 128MB
----------
## Big data file formats - Parquet, ORC, Avro
![](https://paper-attachments.dropbox.com/s_86CFF5479D5C05A018B2538409D8E30C1596B9FEC4E9972D3F9F51C0771A30C2_1586854501502_image.png)

![](https://paper-attachments.dropbox.com/s_86CFF5479D5C05A018B2538409D8E30C1596B9FEC4E9972D3F9F51C0771A30C2_1582806432798_image.png)

- Storing data as **.csv** or **.json** is terribly inefficient
- If you have **nested structure**, then Parquet is more efficient than ORC
- Rule of thumb - Parquet when working with Spark, ORC when working with Hive
----------
## Compression - gzip, snappy, zstd
![https://blog.root.cz/x14/porovnani-gzip-bz2-xz-a-zstd-ve-vsech-stupnich-komprese/](https://paper-attachments.dropbox.com/s_86CFF5479D5C05A018B2538409D8E30C1596B9FEC4E9972D3F9F51C0771A30C2_1582807041940_image.png)

- **zlib, bz2, zstd, xz, gzip, snappy**, …
- **snappy** is the default compression in Spark
- **zstd** is a good compromise in most situations and works well even for small data (e.g. sending JSON documents over network)
- snappy ****compression can efficiently work with large files in Spark, unlike gzip which has to load everything into memory first
----------
## SQL databases - BigQuery, Redshift, Clickhouse, Snowflake, Vertica
- **BigQuery** - pay-per-query, support partitions, **super fast queries**
- **Redshift** - pay per running cluster, supports partitions, hash keys / distributed keys. Can be combined with **Redshift Spectrum** or **Athena** if you store data in S3
- **Azure Synapse Analytics** - both pay-per-query and provisioned cluster. Similar pricing to AWS / GCP, but claims to be faster (on peta-byte scale queries)
- **Clickhouse** - open-source (self-hosted only), easy to set up and use, impressive performance
- **Vertica** - self-hosted community edition
----------
## A practical example
https://colab.research.google.com/drive/1mX9OJd7fCa2eGJN4A4xYkdE-USqYyKr0

----------
# Introduction to Spark
- “**Apache Spark™** is a unified analytics engine for large-scale data processing.”
- “**Run workloads 100x faster.” …** than Hadoop, which could still be really slow
- **In-memory processing**
- Spark ecosystem - SQL, DataFrames, Streaming, GraphX, MLLib
- supports Scala, Java, Python and R
- Version 3.0.1 and 2.5.4 are stable (3.x.x provides performance enhancements, but the API is almost the same)

Even though Spark promises to be fast, there’s still overhead compared to well written parallelized ETL script. The biggest advantage is that it’s a universal tool for almost all big data needs.

----------
## Hello World! in Spark
    # Spark session is a new entry point from Spark 2.0
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    # get Spark context which is an entry gate of Spark functionality
    sc = spark.sparkContext

    # distribute your data across cluster
    num_partitions = 5
    rdd = sc.parallelize(['Hello'] * 1000, num_partitions)
    # >> ['Hello', ..., 'Hello']

    # run heavy computation on your workers
    rdd = rdd.map(lambda x: x + ' World!')
    # >> ['Hello World!', ..., 'Hello World!']

    # count number of elements on all partitions
    rdd = rdd.mapPartitions(lambda x: [len(list(x))])
    # >> [200, 200, 200, 200, 200]

    # return counts from partitions
    # start the computation!
    print(rdd.collect())
    # >> [200, 200, 200, 200, 200]


----------
## GroupBy Example
![GroupBy example](https://backtobazics.com/wp-content/uploads/2015/12/apache-spark-groupby-example.gif)




    # Create RDD with 3 partitions
    # Driver distributes data to nodes
    x = sc.parallelize(["Joseph", "Jimmy", "Tina",
                        "Thomas", "James", "Cory",
                        "Christine", "Jackeline", "Juan"], 3)

    # Apply groupBy operation on x
    # Data is reshuffled between nodes by their first letter
    y = x.groupBy(lambda word: word[0])

    # Collect data from nodes to driver
    for t in y.collect():
        print((t[0],[i for i in t[1]]))


----------
## Directed Acyclic Graph (DAG)
![Word count DAG example](https://databricks.com/wp-content/uploads/2015/06/Screen-Shot-2015-06-19-at-2.00.59-PM.png)

- Spark internally constructs directed acyclic graph representing computation
- Spark Driver has web UI on **port 4040** with all kinds of information
- Definitely use it to understand what’s happening and **don’t run your jobs blindly**
- To access the web UI you have to set up port forwarding through SSH tunnel to your server on port 4040 with
    ssh -NL localhost:4040:localhost:4040 hadoop@ec2-###-##-##-###.compute-1.amazonaws.com
    and then access it on http://localhost:4040/




----------
## Spark Architecture
![Master & slave architecture of Spark](https://paper-attachments.dropbox.com/s_86CFF5479D5C05A018B2538409D8E30C1596B9FEC4E9972D3F9F51C0771A30C2_1585639992916_image.png)

- **Driver** converts the code into DAG (directed acyclic graph) and into physical execution plan with set of **stages**
- **Spark Context** is an access point to the cluster through a resource manager
- **Executor** lives on the slave node and processes **Tasks** from execution plan
- **Cluster Manager** - Standalone, Apache Mesos, Hadoop Yarn, Kubernetes


----------
## RDDs (Resilient Distributed Datasets)
![](https://paper-attachments.dropbox.com/s_86CFF5479D5C05A018B2538409D8E30C1596B9FEC4E9972D3F9F51C0771A30C2_1586857343549_image.png)

- **Collection of elements** that can be operated on in parallel
- Can be created by:
    - parallelizing data on driver - cuts your data to N partitions
    - reference external storage filesystem such as HDFS


----------
## DataFrames
![Spark structures](https://i.stack.imgur.com/3rF6p.png)

- **DataFrame** is a 2-dimensional **labeled data structure** with columns of potentially different types. You can think of it like a spreadsheet or SQL table
- DataFrames in Spark are an effective way for working with **structured data**
- DataSet is “strongly typed RDD”
- Similar to Pandas, can be easily converted back and forth (if it fits into memory)

    import pandas as pd
    df = pd.DataFrame({
      'a': [1, 2, 3],
      'b': ['a', 'b', 'c'],
    })
    sf = spark.createDataFrame(df)
    df = sf.toPandas()


----------
## How to run Spark
****- **Driver & workers locally**
    - not that useful (besides testing and prototyping), that would mean you don’t have big data
- **Driver & workers remotely and sending requests through Livy**
    - interactive analysis in Jupyter
    - [sparkmagic](https://github.com/jupyter-incubator/sparkmagic)
    - Spark kernel https://toree.apache.org/
- **Driver locally & cluster of workers**
- **AWS**
    - Run Spark on Elastic Map Reduce
    - or AWS Glue which is fully-managed, pay-per-usage ETL service with Spark inside
    - Connect to Spark from Sagemaker notebook
- **Google Cloud**
    - Dataproc
- **Databricks**
    - managed Spark with their own notebooks, also check out its open-source “clone” https://zeppelin.apache.org/
----------
## Spot instances
- **Always use spot instances** (preemptive instances) for **worker nodes** for up to 90% cost savings (in reality it’s more like 60%)
- Otherwise there’s little point in having Spark cluster compared to having larger workstation as the price scales linearly with CPU & memory
- You should have one on-demand master node and scale your workers based on your computation needs to save money (i.e. add workers in the morning, kill them after work)
- Choose between memory optimized and CPU optimized instances based on what is the bottleneck


----------
## Most common Spark mistakes

**Your data doesn’t fit into total workers memory**
Make sure your data size is smaller than total cluster memory in **all stages** (e.g. joins). Think about the DAG, pre-filter data before joins, etc.

**Imbalanced partitions**
When doing groupby, all groups should have similar size. If partition is too large it may not fit into node memory.

**Exploding joins**
Can happen if you have duplicate items in both A and B, then the resulting table grows quadratically.

**Insufficient partitioning**
More partitions = smoother job without errors (with a bit more overhead).

**Unnecessary shuffling**
Shuffling between nodes is costly, try to come up with a better way.

**Data compressed in huge gzip files**
Spark must first ungzip them (thus hold it all in memory) and only then you can partitoin it and filter. It’s better to use parquet or snappy.

**Ignoring corrupted records**
Function `spark.read.csv` has argument `mode` with these options

- PERMISSIVE (default) - create column `columnNameOfCorruptRecord` if malformed record is found and use null for others
- DROPMALFORMED - ignores corrupted records
- FAILFAST - throws an exception on corrupted records

People tend to ignore the corrupted columns and work with null values, it’s better to use FAILFAST for production and PERMISSIVE for debugging

----------
## Alternatives - Apache Beam (Dataflow), Dask, lambdas
- **Apach Beam (Dataflow)**
    - unified **batch and streaming**
    - Dataflow is pay-per-usage with no setup and infinite scalability
- **Dask**
    - integrates well with python ecosystem (numpy, pandas, scikit-learn)
    - easier to setup and to use compared to PySpark
    - mostly used to scale computations locally across cores
- **Lambda functions**
    - cheap and easy way to do linear ETL
- **Larger machine**
    - there’s no shame in using multiprocessing or counting words with `wc`
----------
## A practical example
https://colab.research.google.com/drive/1gF_v3p_KyArAfiimqhpJ09ySTpDDtlzs

----------
# ML strategies for Big Data
----------
## Incremental learning
![](http://rasbt.github.io/mlxtend/user_guide/general_concepts/gradient-optimization_files/ball.png)


[http://rasbt.github.io/mlxtend/user_guide/general_concepts/gradient-optimization_files/ball.png](http://rasbt.github.io/mlxtend/user_guide/general_concepts/gradient-optimization_files/ball.png)

- Some algorithms don’t need to see all training examples at once and can be trained incrementaly on small batches of data
- **Gradient descent** is an optimization method for finding local minima of differentiable loss functions
- For example linear regression minimizes loss function $$J(\beta) = \frac{1}{2} \sum_{x_i} (y_i - \beta x_i)^2$$ which has a derivative $$\nabla J(\beta) = \sum_{x_i} (y_i - \beta x_i) x_i$$. Then the gradient descent looks like$$\beta_{t+1} = \beta_{t} - \gamma \nabla J(\beta_t) = \beta_{t} - \gamma \sum_{x \in batch} (y_i - \beta_t x_i) x_i$$
- **Examples**: multivariate or logistic regression, neural networks, PCA, factorization machines, matrix factorization for recommendations, …
- Some **scikit-learn** algorithms implement `.partial_fit` method for incremental learning
- Algorithms like decision trees need all training data for the split, although it can be fixed by smart heuristics
- Naturally supports streaming data and **online learning**

----------
## Batch learning for neural networks
- NN allows incremental learning through gradient descent
- If your data don’t fit into memory, read it chunk by chunk from disk or distributed filesystem with `tf.data` from Tensoflow such as `tf.data.Dataset.from_generator`, `tf.data.TextLineDataset` or `tf.data.experimental.make_csv_dataset`
- Make sure to shuffle your dataset with `Dataset.shuffle()` to prevent overfitting on batches
- Supports HDFS distributed files, but the bandwidth can be a bottleneck, try to `Dataset.interleave` to read in parallel

----------
## Distributed training

Two training strategies:

1. **Synchronous - e**ach worker trains on its own slice of data, then we aggregate gradients from workers
2. **Asynchronous -** workers are independently training and updating parameters asynchornously
![Synchronous (a) and asynchronous (b) distributed training](https://www.researchgate.net/profile/Kun_Qiu2/publication/320319257/figure/fig1/AS:631653764444161@1527609401039/Synchronous-and-asynchronous-path-computation-In-synchronous-computation-the.png)

- Tensorflow offers different types of distributed training (the best depends on your topology and CPU-GPU vs GPU-GPU communication)
    - `tf.distribute.MirroredStrategy`
    - `tf.distribute.CentralStorageStrategy`
    - …
- well abstracted in TF

    strategy = tf.distribute.MirroredStrategy()

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    with strategy.scope():
      model = tf.keras.Sequential([...
----------
## Federated learning
- Train an algorithm on data **without ever seeing the actual data points**
- Needed when you can’t get data out of the device because of privacy and security
    - training personalization algorithm on mobile phones
    - aggregating medical data from several hospitals

![Federated learning general process in central orchestrator setup](https://upload.wikimedia.org/wikipedia/commons/e/e2/Federated_learning_process_central_case.png)

----------
## Random sampling (“training data compression”)
- Select each sample with probability $$p(x_i)$$ and then use **sample weight** $$\frac{1}{p(x_i)}$$in the algorithm
- Let’s say you have 10,000 non-clicks (views), 100 clicks and you are estimating click-through-rate
    - Select non-clicks with 10% probability → ~1000 non-clicks with weight 10
    - Select clicks with 100% probability → 100 clicks with weight 1
    - We **reduced sample size by 89%** yet retained all positive cases
- Trick used by Google, Facebook, Baidu when working with huge datasets
- Use deterministic sampling to avoid biases (e.g. `(hash(sample.id) % 10000) * 10000.`)

**Object Importance**

- Calculate the effect of objects from the training dataset on the optimized metric values for the objects from the validation dataset (supported by CatBoost)
- Open question:
        “*Can we use our model to calculate object importance, caculate it for entire training data and use it as* $$p(x_i)$$ *to reduce data size without losing its predictive power?*“

----------
## Alternative strategies

**Submodels**

- Split your data by feature with the most discriminative power (e.g. feature from the first split of decision tree or the one with the highest mutual information with data)
- Advantages:
    - faster training and possibly faster prediction time
    - manage smaller models (easier debugging)
    - can have even better performance if splits by feature have different characteristics
- Disadvantege:
    - manage more models
    - worse generalization


**Larger workstation**

- Workstation with 1TB memory costs ~$7/hour
- You could prototype your model on subset of data (random sampling) and then retrain model periodically on huge machine
- In the end could be easier & faster & cheaper than having to manage Spark cluster

----------
## Frameworks

**MLlib**

- See Colab notebook below

**Scikit-learn with** ***partial_fit***

- See Incremental learning and Colab notebook below

**Distributed XGBoost**

- Boosting cannot be trivially parallelized, so XGboost gets locally best splits from all workers (each having subset of features and data) and then chooses the best global split.
- Chosen split is not guaranteed to be globally optimal, but on the other hand it works as a kind of regularization

**Dask-ML**

- very immature project, but it's gaining some momentum
- natively supports only generalized linear models and some clustering methods
- **integration with XGBoost**


----------
## Running Spark ML model in production
- Serving trained models in production has high overhead since we have to create Spark context and Spark DataFrame for prediction
- https://mleap-docs.combust.ml/ promises single-digit ms latency (pure Spark has ~50ms overhead at the time of writing), but it’s not up to date with all transformers
- https://www.mlflow.org/ or Sagemaker can be used for easy managing & serving models with MLeap or Spark
----------
## Practical examples
https://colab.research.google.com/drive/1Wgz1sXd0dBpCgFRN3LmzF4Yhay--y_ji

https://colab.research.google.com/drive/1P19JculqNRAZgX7tsRdMAmUMlJZ4hkAS

https://colab.research.google.com/drive/1XsYH9wfTJDjMmwlpbdg5HrjFnkKJ3mft

----------
# **Training & Deploying MLLib model on cluster**
## AWS

Export notebook with instructions from [Colab notebook](https://colab.research.google.com/drive/1IUBJ9fORNofsIPm31kbuOTlUYSnOjiz-) to Jupyter notebook. You can’t run it on Colab, but must run it locally in Jupyter.


## Azure

Check notebook with instructions from [Colab notebook](https://colab.research.google.com/drive/1fTc5DoJanKSLHNdpAuguKpW3SsfBz4pY?usp=sharing).


----------
#                      Thank you! Questions?
----------
# List of notebooks
- [Data size and model capacity](https://colab.research.google.com/drive/1mX9OJd7fCa2eGJN4A4xYkdE-USqYyKr0)
- [Spark Introduction](https://colab.research.google.com/drive/1gF_v3p_KyArAfiimqhpJ09ySTpDDtlzs)
- [Spark MLLib](https://colab.research.google.com/drive/1Wgz1sXd0dBpCgFRN3LmzF4Yhay--y_ji)
- [Dask & Dask-ML](https://colab.research.google.com/drive/1P19JculqNRAZgX7tsRdMAmUMlJZ4hkAS)
- [Scikit-learn incremental learning](https://colab.research.google.com/drive/1XsYH9wfTJDjMmwlpbdg5HrjFnkKJ3mft#scrollTo=94i_3y36Qe71)
- [MLLib model on AWS EMR cluster](https://colab.research.google.com/drive/1IUBJ9fORNofsIPm31kbuOTlUYSnOjiz-)
- [MLLib model on Azure HDINsight cluster](https://colab.research.google.com/drive/1fTc5DoJanKSLHNdpAuguKpW3SsfBz4pY?usp=sharing)


## Notebook solutions
- [Data size and model capacity](https://colab.research.google.com/drive/1YsebNjMYVaPZLviWGeUJWvbl9aPYX0rk?usp=sharing)
- [Spark Introduction](https://colab.research.google.com/drive/1il_q6yNLdTtpOv_JUx0pP5pnheYHzldn?usp=sharing)
- [Spark MLLib](https://colab.research.google.com/drive/1v6Nkd4qV75ZmOjoygqLVckSi0yKVwjOT?usp=sharing)



# Supplementary materials
## Workflow orchestration

**Apache Airflow**

- best tool for ETL, has the largest community, good for ML pipelines too
- non-trivial to setup, but easy to maintain after
- large number of plugins (kubernetes, clouds, …)
![](https://miro.medium.com/max/4800/1*w_SwwTYbpmBB9-IATyqP9A.png)


**Luigi**

- Python-focused, steeper learning curve than Airflow
- Steps are classes and each run generates some output (preprocessed data, model, diagnostic report, ...)
- Easy setup, can be run locally, on server, from code...
- Parametrization and parallel processing (e.g. task can have parameter `country` and generate one file for each country)
![](https://res.cloudinary.com/practicaldev/image/fetch/s--pL5uk-vt--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://miro.medium.com/max/2192/1%2AhURwKOAd15U2xEkZeXVrfQ.png)


**Metaflow**

> Metaflow is a human-friendly Python/R library that helps scientists and engineers build and manage real-life data science projects. Metaflow was originally developed at Netflix to boost productivity of data scientists who work on a wide variety of projects from classical statistics to state-of-the-art deep learning.

- Works only on AWS
- Simple and intuitive interface, easy to use by both engineers and data scientists
- The ability to resume a flow, re-executing all successful steps (great for debugging)

**MLFLow Projects**

- training data preparation exclusively for ML
- easy to use, but hard to setup more complex pipelines

**Notebooks**

- pioneered by Netflix, notebooks serve as “jobs”
- notebook parametrization by Papermill, scheduling by Luigi / Airflow
- easy debugging & monitoring


## MLOps

= Productionizing ML models

![](https://www.xenonstack.com/wp-content/uploads/2019/09/mlops-solutions-services-xenonstack.png)


**Options?**

- Custom solution
    - Docker container with trained model saved on disk (or in S3 / HDFS)
    - FastAPI / Flask for serving predictions
    - Easiest option, can be just added to existing python REST services
- Sagemaker (AWS)
    - hard to setup, not flexible enough to be customized
    - evolving quickly, good for later phase of the project
- MLFlow
    - open-source, not so intuitive
    - management overhead (do you really need all the features?)

