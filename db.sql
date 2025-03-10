drop database if exists video_dataset;
create database video_dataset;
use video_dataset;

create table users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(50), 
    password VARCHAR(50),
    username VARCHAR(20),
    age INT,
    gender VARCHAR(10)
    );
