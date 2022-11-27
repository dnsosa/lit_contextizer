<?php

$servername = "<-- MYSQL LOGIN INFO -->";
$username = "<-- MYSQL LOGIN INFO -->";
$password = "<-- MYSQL LOGIN INFO -->";
$dbname = "<-- MYSQL LOGIN INFO -->";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
$conn->set_charset("utf8");

?>