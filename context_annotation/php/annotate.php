<?php

include 'db_connect.php';

$jobdone = false;
if (isset($_GET['relation_id'])) {

	$relation_id = $_GET['relation_id'];
	if (!is_numeric($relation_id))
		exit();

} else {
	$sql = "SELECT relation_id FROM relations WHERE notes IS NULL AND relation_id NOT IN (SELECT relation_id FROM annotations)";
	$result = $conn->query($sql);
	if (!$result) {
		$error = mysqli_error($conn);
		echo "<p>ERROR $error with SQL: $sql</p>";
		exit(1);
	}
	
	if ($result->num_rows > 0) {
		$row = $result->fetch_assoc();
		$relation_id = $row['relation_id'];
	} else {
		$jobdone = true;
	}
}

if (!$jobdone) {
	$sql = "SELECT r.sentence, r.entity1, r.entity2, r.index_in_doc, r.total_in_doc, r.notes, d.document_id, d.pmid, d.pmcid, d.doi, d.title, d.journal, d.year FROM relations r, documents d WHERE r.relation_id = '$relation_id' AND r.document_id = d.document_id LIMIT 1";

	$contexts = [];
	$existing_annotations = [];

	$result = $conn->query($sql);
	if ($result->num_rows > 0) {
		$jobdone = false;
		$data = $result->fetch_assoc();
		
		$sentence = $data['sentence'];
		$entity1 = $data['entity1'];
		$entity2 = $data['entity2'];
		$index_in_doc = $data['index_in_doc'];
		$total_in_doc = $data['total_in_doc'];
		$notes = $data['notes'];
		$document_id = $data['document_id'];
		$pmid = $data['pmid'];
		$pmcid = $data['pmcid'];
		$doi = $data['doi'];
		$title = $data['title'];
		$journal = $data['journal'];
		$year = $data['year'];
		
		$sentence = str_replace('<entity1>','<span style="font-weight: bold; color: #3c78d8;">',$sentence);
		$sentence = str_replace('</entity1>','</span>',$sentence);
		$sentence = str_replace('<entity2>','<span style="font-weight: bold; color: #674ea7;">',$sentence);
		$sentence = str_replace('</entity2>','</span>',$sentence);
		
		$sql = "SELECT context_id, source_context_id, name, type FROM contexts WHERE document_id = '$document_id' ORDER BY type,name";
		$result = $conn->query($sql);
		while ($row = $result->fetch_assoc()) {
			$context_id = $row['context_id'];
			$source_context_id = $row['source_context_id'];
			$name = $row['name'];
			$type = $row['type'];
			
			if (!array_key_exists($type, $contexts))
				$contexts[$type] = [];
				
			$contexts[$type][] = array("context_id"=>$context_id, "source_context_id"=>$source_context_id, "name"=>$name, "type"=>$type);
		}
		
		$sql = "SELECT context_id FROM annotations WHERE relation_id = '$relation_id'";
		$result = $conn->query($sql);
		while ($row = $result->fetch_assoc()) {
			$existing_annotations[] = $row['context_id'];
		}
	} else {
		$jobdone = true;
	}
}

//$sql = "SELECT COUNT(DISTINCT(a.relation_id)) as count FROM annotations a, relations r WHERE a.relation_id = r.relation_id";
$sql = "SELECT COUNT(DISTINCT(document_id)) as count FROM relations WHERE notes IS NOT NULL OR relation_id IN (SELECT relation_id FROM annotations)";

$result = $conn->query($sql);
if (!$result) {
	$error = mysqli_error($conn);
	echo "<p>ERROR $error with SQL: $sql</p>";
	exit(1);
}
if ($result->num_rows > 0) {
	$row = $result->fetch_assoc();
	$annotated_count = $row['count'];
} else {
	$annotated_count = 0;
}

if ($annotated_count == 1)
	$annotated_count_text = "1 document annotated";
else
	$annotated_count_text = "$annotated_count documents annotated";

if (isset($_GET['savedmsg']))
	$savedmsg = htmlspecialchars($_GET['savedmsg']);
else
	$savedmsg = null;
	

?>
<!doctype html>
<html lang="en">
	<head>
		<!-- Required meta tags -->
		<meta charset="utf-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1" />

		<!-- Bootstrap CSS -->
		<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-giJF6kkoqNQ00vy+HMDP7azOuL0xtbfIcaT9wjKHr8RbDVddVHyTfAAsrekwKmP1" crossorigin="anonymous" />

		<style>
			.bd-placeholder-img {
			font-size: 1.125rem;
			text-anchor: middle;
			-webkit-user-select: none;
			-moz-user-select: none;
			user-select: none;
			}

			@media (min-width: 768px) {
			.bd-placeholder-img-lg {
			font-size: 3.5rem;
			}
			}
		</style>


		<title>Context-annotation</title>
	</head>
	<body>


		<nav class="navbar navbar-expand-md navbar-dark bg-dark mb-4">
			<div class="container-fluid">
				<a class="navbar-brand" href="index.php">Context-annotation</a>
				<button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarCollapse" aria-controls="navbarCollapse" aria-expanded="false" aria-label="Toggle navigation">
					<span class="navbar-toggler-icon"></span>
				</button>
				<div class="collapse navbar-collapse" id="navbarCollapse">
					<ul class="navbar-nav me-auto mb-2 mb-md-0">
						<li class="nav-item">
							<a class="nav-link" aria-current="page" href="index.php">View Annotations</a>
						</li>
						<li class="nav-item">
							<a class="nav-link active" aria-current="page" href="annotate.php">Annotate</a>
						</li>
					</ul>
				</div>
				
				<div>
					<ul class="navbar-nav">
						<li class="nav-item" style="width: 200px; text-align: right;">
							<a class="nav-link" aria-current="page" href="index.php"><?php echo $annotated_count_text ?></a>
						</li>
					</ul>
				</div>
				
			</div>
			
			
		</nav>

		<main class="container">
			<div class="container">
			
				<?php if (!is_null($savedmsg)) { ?>
				<div class="alert alert-warning alert-dismissible fade show" role="alert">
					<strong>Saved!</strong> <?php echo $savedmsg ?>
					<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
				</div>
				<?php } ?>
			
			<?php if (!$jobdone) { ?>
				<div class="bg-light p-5 rounded" style="height:100%">
					<p class="lead" style="font-family: Arial, Helvetica, sans-serif"><?php echo $sentence ?></p>
					<p>
						<span style="font-style: italic;"><?php echo "$title ($journal) [$index_in_doc/$total_in_doc]" ?> </span>
						<br />
						<a href="http://doi.org/<?php echo $doi ?>" target="_blank">DOI</a>
						<a href="https://pubmed.ncbi.nlm.nih.gov/<?php echo $pmid ?>" target="_blank">PubMed</a>
						<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC<?php echo $pmcid ?>/" target="_blank">PMC</a>
						<a href="https://www.ncbi.nlm.nih.gov/research/pubtator/?view=docsum&query=<?php echo $pmid ?>" target="_blank">PubTator</a>
					</p>
				</div>
			<?php } else { ?>
				<div class="bg-light p-5 rounded" style="height:100%">
					<p class="lead" style="font-family: Arial, Helvetica, sans-serif">No more relations to annotate</p>
				</div>
			
			<?php } ?>
			</div>

			<!-- <p class="lead" style="font-family: Arial, Helvetica, sans-serif"><span style="font-weight: bold; color: #3c78d8;">Mec1/ATR</span> inhibits mitotic <span style="font-weight: bold; color: #674ea7;">Cdk1</span> activity through downstream effector kinases, Rad53/Chk2 and Swe1/Wee1.</p> -->

			<?php if (!$jobdone) { ?>
			<form action="add_annotation.php" method="post">
				<input type="hidden" name="relation_id" value="<?php echo $relation_id; ?>" />
				<input type="hidden" name="entity1" value="<?php echo $entity1; ?>" />
				<input type="hidden" name="entity2" value="<?php echo $entity2; ?>" />
				<div class="row" style="margin-top:20px">
					<div class="col-sm-7" style="text-align: center;">
							
							<?php
								foreach ($contexts as $type => $entities) {
										echo "<p>$type: ";
										foreach ($entities as $entity) {
											$context_id = $entity['context_id'];
											$source_context_id = $entity['source_context_id'];
											$name = $entity['name'];
											
											$style = 'info';
											if ($type == 'Disease')
												$style = 'danger';
											elseif ($type == 'Species')
												$style = 'success';
											elseif ($type == 'Chemical')
												$style = 'info';
											elseif ($type == 'CellLine')
												$style = 'warning';
											
											$is_checked = in_array($context_id,$existing_annotations) ? ' checked' : '';
											
											echo "<input name='context_$context_id' type='checkbox' class='btn-check' id='btn-check-$context_id' autocomplete='off' $is_checked/>\n";
											echo "<label class='btn btn-outline-$style' style='margin-bottom:4px' for='btn-check-$context_id'>$name</label>\n";
										}
										echo "</p>";
								}
							?>
					</div>
					<div class="col-sm-3">
							<div class="mb-2">
								<textarea name="notes" class="form-control" id="annotation_notes" rows="2" placeholder="Notes" ><?php echo $notes; ?></textarea>
							</div>
						


					</div>
						<!-- <button type="submit" class="btn btn-warning">Prev</button> -->
						<!-- <button type="submit" class="btn btn-warning">Next</button> -->
					<div class="col-sm-1">
						<button type="submit" class="btn btn-primary">Save &amp; Next</button>
					</div>
					<div class="col-sm-1">
					</div>
				</div>
				<div style="text-align: center;">
				</div>
			</form>
			<?php } ?>


		</main>


		<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>

		<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>

		<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"></script>
	
	</body>
</html>
