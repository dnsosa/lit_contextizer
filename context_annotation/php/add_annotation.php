<?php

include 'db_connect.php';

$relation_id = $_POST['relation_id'];
if (!is_numeric($relation_id))
	exit();
	

$sql = "DELETE FROM annotations WHERE relation_id = '$relation_id'";
$result = $conn->query($sql);
if (!$result) {
	$error = mysqli_error($conn);
	echo "<p>ERROR $error with SQL: $sql</p>";
	exit(1);
}

$annotations = [];
foreach ($_POST as $key => $value) {
	if (substr($key, 0, 8) === "context_") {
		$id = substr($key,8);
		if (is_numeric($id)) {
			$annotations[] = $id;
		}
	}
}

$annotations = array_unique($annotations);

foreach ($annotations as $annotation) {
	$sql = "INSERT INTO annotations(relation_id,context_id,added) VALUES('$relation_id','$annotation',NOW())";
	$result = $conn->query($sql);
	
	if (!$result) {
		$error = mysqli_error($conn);
		echo "<p>ERROR $error with SQL: $sql</p>";
		exit(1);
	}
}


$notes = mysqli_real_escape_string($conn, $_POST['notes']);
$sql = "UPDATE relations SET notes='$notes' WHERE relation_id = '$relation_id'";
$result = $conn->query($sql);
if (!$result) {
	$error = mysqli_error($conn);
	echo "<p>ERROR $error with SQL: $sql</p>";
	exit(1);
}


$entity1 = $_POST['entity1'];
$entity2 = $_POST['entity2'];

$savedmsg = urlencode("$entity1 / $entity2");

header("Location:annotate.php?savedmsg=$savedmsg");
	
?>