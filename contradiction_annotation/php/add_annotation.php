<?php

include 'db_connect.php';

/*echo "<pre>";
print_R($_POST);
echo "</pre>";*/

$pair_id = $_POST['pair_id'];
if (!is_numeric($pair_id))
	exit();

$sql = "DELETE FROM annotations WHERE pair_id = '$pair_id'";
$result = $conn->query($sql);
if (!$result) {
	$error = mysqli_error($conn);
	echo "<p>ERROR $error with SQL: $sql</p>";
	exit(1);
}

$annotations = [];
foreach ($_POST as $key => $value) {
	if (substr($key, 0, 7) === "option_") {
		$id = substr($key,7);
		if (is_numeric($id)) {
			$annotations[] = $id;
		}
	}
}

if (isset($_POST['new_option']) && $_POST['new_option'] != '') {
	$newoptions = explode(',',$_POST['new_option']);
	foreach ($newoptions as $newoption) {
		$newoption = mysqli_real_escape_string($conn,trim($newoption));
		$sql = "SELECT option_id,name FROM options WHERE name='$newoption'";
		$result = $conn->query($sql);
		if (!$result) {
			$error = mysqli_error($conn);
			echo "<p>ERROR $error with SQL: $sql</p>";
			exit(1);
		}
		
		if ($result->num_rows > 0) {
			$row = $result->fetch_assoc();
			$annotations[] = $row['option_id'];
		} else {
			
			$sql = "INSERT INTO options(name) VALUES('$newoption')";
			$result = $conn->query($sql);
			if (!$result) {
				$error = mysqli_error($conn);
				echo "<p>ERROR $error with SQL: $sql</p>";
				exit(1);
			}
			$new_id = mysqli_insert_id($conn);
			$annotations[] = $new_id;
		}
	}
}

$annotations = array_unique($annotations);

foreach ($annotations as $annotation) {
	$sql = "INSERT INTO annotations(pair_id,option_id,added) VALUES('$pair_id','$annotation',NOW())";
	$result = $conn->query($sql);
	
	if (!$result) {
		$error = mysqli_error($conn);
		echo "<p>ERROR $error with SQL: $sql</p>";
		exit(1);
	}
}


$notes = mysqli_real_escape_string($conn, $_POST['notes']);
$sql = "UPDATE pairs SET notes='$notes' WHERE pair_id = '$pair_id'";
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