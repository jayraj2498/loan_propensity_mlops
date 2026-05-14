# ─────────────────────────────────────────────────────────────
# infrastructure/terraform/main.tf
# Provisions AWS infrastructure:
#   - S3 bucket (data + model storage)
#   - ECR repository (Docker images)
#   - EC2 instance (API hosting)
#   - Security groups
# ─────────────────────────────────────────────────────────────

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # Store Terraform state in S3 (enables team collaboration)
  backend "s3" {
    bucket = "loan-propensity-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "ap-south-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Variables ─────────────────────────────────────────────────
variable "aws_region"    { default = "ap-south-1" }
variable "project_name"  { default = "loan-propensity-mlops" }
variable "instance_type" { default = "t3.medium" }

# ── S3 bucket: data + model artifacts ────────────────────────
resource "aws_s3_bucket" "mlops_bucket" {
  bucket = "${var.project_name}-${random_id.suffix.hex}"
  tags   = { Name = var.project_name, Environment = "prod" }
}

resource "aws_s3_bucket_versioning" "mlops_bucket_versioning" {
  bucket = aws_s3_bucket.mlops_bucket.id
  versioning_configuration { status = "Enabled" }
}

resource "random_id" "suffix" { byte_length = 4 }

# ── ECR repository: Docker images ─────────────────────────────
resource "aws_ecr_repository" "api_repo" {
  name                 = "${var.project_name}-api"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

# ── Security group: allow HTTP + SSH ─────────────────────────
resource "aws_security_group" "api_sg" {
  name        = "${var.project_name}-sg"
  description = "Allow inbound traffic to loan propensity API"

  ingress {
    description = "FastAPI"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "Streamlit"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ── EC2 instance ──────────────────────────────────────────────
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

resource "aws_instance" "api_server" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.instance_type
  vpc_security_group_ids = [aws_security_group.api_sg.id]

  user_data = <<-EOF
    #!/bin/bash
    yum update -y
    yum install -y docker
    systemctl start docker
    systemctl enable docker
    usermod -aG docker ec2-user
    yum install -y aws-cli
  EOF

  tags = {
    Name        = "${var.project_name}-api-server"
    Environment = "prod"
  }
}

# ── Outputs ───────────────────────────────────────────────────
output "ec2_public_ip"    { value = aws_instance.api_server.public_ip }
output "ecr_repository_url"{ value = aws_ecr_repository.api_repo.repository_url }
output "s3_bucket_name"   { value = aws_s3_bucket.mlops_bucket.bucket }
