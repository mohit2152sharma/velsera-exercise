resource "aws_eip" "nat" {
  domain = "vpc"

  tags = {
    Name        = "nat"
    application = "velsera"
  }
}

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name        = "nat"
    application = "velsera"
  }

  depends_on = [aws_internet_gateway.igw]
}
