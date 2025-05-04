resource "aws_iam_role" "velsera" {
  name = "eks-cluster-velsera"

  assume_role_policy = <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "eks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
POLICY
}

resource "aws_iam_role_policy_attachment" "velsera-AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.velsera.name
}

resource "aws_eks_cluster" "velsera" {
  name     = var.cluster_name
  role_arn = aws_iam_role.velsera.arn

  vpc_config {
    subnet_ids = concat(
      [for x in aws_subnet.public : x.id],
      [for x in aws_subnet.private : x.id]
    )
  }

  depends_on = [aws_iam_role_policy_attachment.velsera-AmazonEKSClusterPolicy]
}
