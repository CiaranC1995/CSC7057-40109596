-- phpMyAdmin SQL Dump
-- version 5.1.2
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: Aug 15, 2023 at 10:45 AM
-- Server version: 5.7.24
-- PHP Version: 8.0.1

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `csc7057`
--

-- --------------------------------------------------------

--
-- Table structure for table `classifier_output`
--

CREATE TABLE `classifier_output` (
  `classifier_output_id` int(11) NOT NULL,
  `date_of_classification` text NOT NULL,
  `raw_input_text` text NOT NULL,
  `preprocessed_input_text` text NOT NULL,
  `text_perplexity` double NOT NULL,
  `text_burstiness` double NOT NULL,
  `classifier_output` tinyint(1) NOT NULL,
  `human_probability` double NOT NULL,
  `ai_probability` double NOT NULL,
  `user_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `sentence_perplexity`
--

CREATE TABLE `sentence_perplexity` (
  `sentence_perplexity_id` int(11) NOT NULL,
  `sentence_number` int(11) NOT NULL,
  `sentence` text NOT NULL,
  `perplexity` double NOT NULL,
  `classifier_output_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `user`
--

CREATE TABLE `user` (
  `user_id` int(11) NOT NULL,
  `username` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `password` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Dumping data for table `user`
--

INSERT INTO `user` (`user_id`, `username`, `email`, `password`) VALUES
(-1, 'NoUser', 'nouser@email.co.uk', 'nouser'),
(9, 'ciaranc', 'ciaranc@email.co.uk', '$2b$10$RNiaMuCXXFh23Q0l00.bUOqNUdcoiRonvlERO7fLPzA0mTZA49srm');

-- --------------------------------------------------------

--
-- Table structure for table `user_contact_message`
--

CREATE TABLE `user_contact_message` (
  `user_contact_message_id` int(11) NOT NULL,
  `name` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `subject` varchar(255) NOT NULL,
  `message` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `classifier_output`
--
ALTER TABLE `classifier_output`
  ADD PRIMARY KEY (`classifier_output_id`),
  ADD KEY `FK_user_user_id` (`user_id`);

--
-- Indexes for table `sentence_perplexity`
--
ALTER TABLE `sentence_perplexity`
  ADD PRIMARY KEY (`sentence_perplexity_id`),
  ADD KEY `FK_classifier_output_classifier_output_id` (`classifier_output_id`);

--
-- Indexes for table `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`user_id`);

--
-- Indexes for table `user_contact_message`
--
ALTER TABLE `user_contact_message`
  ADD PRIMARY KEY (`user_contact_message_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `classifier_output`
--
ALTER TABLE `classifier_output`
  MODIFY `classifier_output_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `sentence_perplexity`
--
ALTER TABLE `sentence_perplexity`
  MODIFY `sentence_perplexity_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=18;

--
-- AUTO_INCREMENT for table `user`
--
ALTER TABLE `user`
  MODIFY `user_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=12;

--
-- AUTO_INCREMENT for table `user_contact_message`
--
ALTER TABLE `user_contact_message`
  MODIFY `user_contact_message_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `classifier_output`
--
ALTER TABLE `classifier_output`
  ADD CONSTRAINT `FK_user_user_id` FOREIGN KEY (`user_id`) REFERENCES `user` (`user_id`);

--
-- Constraints for table `sentence_perplexity`
--
ALTER TABLE `sentence_perplexity`
  ADD CONSTRAINT `FK_classifier_output_classifier_output_id` FOREIGN KEY (`classifier_output_id`) REFERENCES `classifier_output` (`classifier_output_id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
